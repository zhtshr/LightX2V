import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.wan.infer.feature_caching.transformer_infer import (
    WanTransformerInferAdaCaching,
    WanTransformerInferCustomCaching,
    WanTransformerInferDualBlock,
    WanTransformerInferDynamicBlock,
    WanTransformerInferFirstBlock,
    WanTransformerInferMagCaching,
    WanTransformerInferTaylorCaching,
    WanTransformerInferTeaCaching,
)
from lightx2v.models.networks.wan.infer.offload.transformer_infer import (
    WanOffloadTransformerInfer,
)
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.models.networks.wan.infer.transformer_infer import (
    WanTransformerInfer,
)
from lightx2v.models.networks.wan.pp_load import distribute_weights_pp_from_rank0
from lightx2v.models.networks.wan.pp_utils import validate_wan_pp_config
from lightx2v.models.networks.wan.tp_load import distribute_weights_from_rank0
from lightx2v.models.networks.wan.tp_utils import validate_wan_tp_config
from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.models.networks.wan.weights.transformer_weights import (
    WanTransformerWeights,
)
from lightx2v.utils.custom_compiler import compiled_method
from lightx2v.utils.envs import *
from lightx2v.utils.utils import *


class WanModel(BaseTransformerModel):
    pre_weight_class = WanPreWeights
    transformer_weight_class = WanTransformerWeights

    def __init__(self, model_path, config, device, model_type="wan2.1", lora_path=None, lora_strength=1.0):
        super().__init__(model_path, config, device, model_type, lora_path, lora_strength)
        if self.lazy_load:
            self.remove_keys.extend(["blocks."])
        self.sensitive_layer = {
            "norm",
            "embedding",
            "modulation",
            "time",
            "img_emb.proj.0",
            "img_emb.proj.4",
            "before_proj",  # vace
            "after_proj",  # vace
        }
        self._init_tp_context()
        self._init_pp_context()
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def _init_tp_context(self):
        if self.config.get("tensor_parallel", False):
            if self.cpu_offload:
                raise NotImplementedError("Wan tensor parallel requires cpu_offload=False")
            validate_wan_tp_config(self.config)
            self.use_tp = True
            self.tp_group = self.config["device_mesh"].get_group(mesh_dim="tensor_p")
            self.tp_rank = dist.get_rank(self.tp_group)
            self.tp_size = dist.get_world_size(self.tp_group)
            self.config["load_from_rank0"] = True
            self.config["_tp_kwargs"] = {
                "tp_group": self.tp_group,
                "tp_rank": self.tp_rank,
                "tp_size": self.tp_size,
            }
            if self.config.get("tp_norm_p2p", False):
                if self.tp_size != 2:
                    raise ValueError("tp_norm_p2p requires tensor_p_size=2")
                from lightx2v.common.ops.norm.tp_p2p_exchange import init_tp_norm_p2p

                init_tp_norm_p2p(self.tp_group)
        else:
            self.use_tp = False
            self.tp_group = None
            self.tp_rank = 0
            self.tp_size = 1
            self.config["_tp_kwargs"] = None

    def _init_pp_context(self):
        if self.config.get("pipeline_parallel", False):
            if self.cpu_offload:
                raise NotImplementedError("Wan pipeline parallel requires cpu_offload=False")
            validate_wan_pp_config(self.config)
            self.use_pp = True
            self.pp_group = self.config["device_mesh"].get_group(mesh_dim="pipe_p")
            self.pp_rank = dist.get_rank(self.pp_group)
            self.pp_size = dist.get_world_size(self.pp_group)
            self.config["pp_rank"] = self.pp_rank
            self.config["load_from_rank0"] = True
        else:
            self.use_pp = False
            self.pp_group = None
            self.pp_rank = 0
            self.pp_size = 1

    def _should_load_weights(self):
        if getattr(self, "use_tp", False) or getattr(self, "use_pp", False):
            if not dist.is_initialized():
                return True
            if getattr(self, "use_pp", False):
                # Each PP stage group has its own leader (pp_rank==0). With PP×SP
                # hybrid, that is one rank per seq_p column, not global rank 0 only.
                return self.pp_rank == 0
            return dist.get_rank() == 0
        return super()._should_load_weights()

    def _load_weights_from_rank0(self, weight_dict, is_weight_loader):
        if getattr(self, "use_pp", False):
            if self.cpu_offload:
                raise NotImplementedError("Wan PP weight distribution requires cpu_offload=False")
            return distribute_weights_pp_from_rank0(
                weight_dict,
                is_weight_loader,
                self.pp_group,
                self.pp_rank,
                self.pp_size,
                self.config["num_layers"],
                self.config.get("pp_layers_per_stage"),
            )
        if self.cpu_offload:
            raise NotImplementedError("Wan TP weight distribution requires cpu_offload=False")
        return distribute_weights_from_rank0(
            weight_dict,
            is_weight_loader,
            self.tp_group,
            self.tp_rank,
            self.tp_size,
        )

    def _apply_weights(self, weight_dict=None):
        if weight_dict is not None:
            self.original_weight_dict = weight_dict
            del weight_dict
            import gc

            gc.collect()

        if getattr(self, "use_pp", False):
            if self.pp_rank == 0:
                self.pre_weight.load(self.original_weight_dict)
            self.transformer_weights.load(self.original_weight_dict)
        else:
            self.pre_weight.load(self.original_weight_dict)
            self.transformer_weights.load(self.original_weight_dict)

        if hasattr(self, "post_weight"):
            self.post_weight.load(self.original_weight_dict)

        if self.config.get("lora_dynamic_apply", False):
            assert self.config.get("lora_configs", False)
            if hasattr(self, "_register_lora"):
                self._register_lora(self.lora_path, self.lora_strength)

        del self.original_weight_dict
        torch.cuda.empty_cache()
        import gc

        gc.collect()

    def _init_infer_class(self):
        self.pre_infer_class = WanPreInfer
        self.post_infer_class = WanPostInfer

        if self.config["feature_caching"] == "NoCaching":
            self.transformer_infer_class = WanTransformerInfer if not self.cpu_offload else WanOffloadTransformerInfer
        elif self.config["feature_caching"] == "Tea":
            self.transformer_infer_class = WanTransformerInferTeaCaching
        elif self.config["feature_caching"] == "TaylorSeer":
            self.transformer_infer_class = WanTransformerInferTaylorCaching
        elif self.config["feature_caching"] == "Ada":
            self.transformer_infer_class = WanTransformerInferAdaCaching
        elif self.config["feature_caching"] == "Custom":
            self.transformer_infer_class = WanTransformerInferCustomCaching
        elif self.config["feature_caching"] == "FirstBlock":
            self.transformer_infer_class = WanTransformerInferFirstBlock
        elif self.config["feature_caching"] == "DualBlock":
            self.transformer_infer_class = WanTransformerInferDualBlock
        elif self.config["feature_caching"] == "DynamicBlock":
            self.transformer_infer_class = WanTransformerInferDynamicBlock
        elif self.config["feature_caching"] == "Mag":
            self.transformer_infer_class = WanTransformerInferMagCaching
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config['feature_caching']}")

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)
        if hasattr(self.transformer_infer, "offload_manager"):
            self._init_offload_manager()

    def _should_init_empty_model(self):
        if self.config.get("lora_configs") and self.config["lora_configs"] and not self.config.get("lora_dynamic_apply", False):
            if self.model_type in ["wan2.1"]:
                return True
            if self.model_type in ["wan2.2_moe_high_noise"]:
                for lora_config in self.config["lora_configs"]:
                    if lora_config["name"] == "high_noise_model":
                        return True
            if self.model_type in ["wan2.2_moe_low_noise"]:
                for lora_config in self.config["lora_configs"]:
                    if lora_config["name"] == "low_noise_model":
                        return True
        return False

    @compiled_method()
    @torch.no_grad()
    def _infer_cond_uncond(self, inputs, infer_condition=True):
        if getattr(self, "use_pp", False):
            return self._infer_cond_uncond_pp(inputs, infer_condition)
        self.scheduler.infer_condition = infer_condition

        pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs)

        if self.config["seq_parallel"]:
            pre_infer_out = self._seq_parallel_pre_process(pre_infer_out)

        x = self.transformer_infer.infer(self.transformer_weights, pre_infer_out)

        if self.config["seq_parallel"]:
            x = self._seq_parallel_post_process(x)

        noise_pred = self.post_infer.infer(x, pre_infer_out)[0]

        if self.clean_cuda_cache:
            del x, pre_infer_out
            torch.cuda.empty_cache()

        return noise_pred

    @torch.no_grad()
    def _infer_cond_uncond_pp(self, inputs, infer_condition=True):
        from lightx2v.models.networks.wan.infer.pipeline_parallel import (
            recv_activation,
            recv_noise_pred,
            recv_pre_metadata,
            send_activation,
            send_noise_pred,
            send_pre_metadata,
        )

        self.scheduler.infer_condition = infer_condition
        last_rank = self.pp_size - 1
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        if self.pp_rank == 0:
            pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs)
            if self.config["seq_parallel"]:
                pre_infer_out = self._seq_parallel_pre_process(pre_infer_out)
            send_pre_metadata(pre_infer_out, last_rank, self.pp_group)
            self.transformer_infer.cos_sin = pre_infer_out.cos_sin
            self.transformer_infer.reset_infer_states()
            x = self.transformer_infer.infer_main_blocks(self.transformer_weights.blocks, pre_infer_out)
            send_activation(x, last_rank, self.pp_group)
            noise_pred = recv_noise_pred(last_rank, self.pp_group, device)
            if self.clean_cuda_cache:
                del x, pre_infer_out
                torch.cuda.empty_cache()
            return noise_pred

        if self.pp_rank == last_rank:
            pre_infer_out = recv_pre_metadata(0, self.pp_group, device)
            x = recv_activation(0, self.pp_group, device)
            pre_infer_out.x = x
            # SF (wan2.1_sf): fill RoPE extras if sender omitted them / local table is enough.
            if getattr(pre_infer_out, "freqs", None) is None and hasattr(self.pre_infer, "freqs"):
                pre_infer_out.freqs = self.pre_infer.freqs
            if getattr(pre_infer_out, "seq_lens", None) is None:
                pre_infer_out.seq_lens = torch.tensor(
                    [x.size(0)], dtype=torch.int32, device=device
                ).unsqueeze(0)
            self.transformer_infer.cos_sin = pre_infer_out.cos_sin
            self.transformer_infer.reset_infer_states()
            x = self.transformer_infer.infer_main_blocks(self.transformer_weights.blocks, pre_infer_out)
            x = self.transformer_infer.infer_non_blocks(
                self.transformer_weights, x, pre_infer_out.embed
            )
            if self.config["seq_parallel"]:
                x = self._seq_parallel_post_process(x)
            noise_pred = self.post_infer.infer(x, pre_infer_out)[0]
            send_noise_pred(noise_pred, 0, self.pp_group)
            if self.clean_cuda_cache:
                del x, pre_infer_out
                torch.cuda.empty_cache()
            return noise_pred

        raise RuntimeError(f"unsupported pp_rank={self.pp_rank} for pp_size={self.pp_size}")

    @torch.no_grad()
    def _stripe_full_q_enabled(self) -> bool:
        """Stripe-KV with unreplicated/full activation (skip seq chunk of x/Q)."""
        import os

        parallel = self.config.get("parallel") or {}
        if not isinstance(parallel, dict):
            return False
        attn_type = str(parallel.get("seq_p_attn_type", ""))
        if attn_type not in ("stripe", "stripe_kv", "ring_kv_cache"):
            return False
        if parallel.get("stripe_full_q"):
            return True
        return os.environ.get("LIGHTX2V_STRIPE_FULL_Q", "0") == "1"

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        # Stripe full-Q mode: keep full activation on every rank; only KV is seq-striped.
        if self._stripe_full_q_enabled():
            return pre_infer_out

        x = pre_infer_out.x
        world_size = dist.get_world_size(self.seq_p_group)
        cur_rank = dist.get_rank(self.seq_p_group)
        padding_size = (world_size - (x.shape[0] % world_size)) % world_size
        if padding_size > 0:
            x = F.pad(x, (0, 0, 0, padding_size))

        pre_infer_out.x = torch.chunk(x, world_size, dim=0)[cur_rank]

        if self.config["model_cls"] in ["wan2.2", "wan2.2_audio"] and self.config["task"] in ["i2v", "s2v", "rs2v"]:
            embed, embed0 = pre_infer_out.embed, pre_infer_out.embed0

            padding_size = (world_size - (embed.shape[0] % world_size)) % world_size
            if padding_size > 0:
                embed = F.pad(embed, (0, 0, 0, padding_size))
                embed0 = F.pad(embed0, (0, 0, 0, 0, 0, padding_size))

            pre_infer_out.embed = torch.chunk(embed, world_size, dim=0)[cur_rank]
            pre_infer_out.embed0 = torch.chunk(embed0, world_size, dim=0)[cur_rank]

        return pre_infer_out

    @torch.no_grad()
    def _seq_parallel_post_process(self, x):
        # Full activation already present on every rank in stripe_full_q mode.
        if self._stripe_full_q_enabled():
            return x

        world_size = dist.get_world_size(self.seq_p_group)
        gathered_x = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(gathered_x, x, group=self.seq_p_group)
        combined_output = torch.cat(gathered_x, dim=0)
        return combined_output

    @torch.no_grad()
    def infer(self, inputs):
        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == 0 and "wan2.2_moe" not in self.config["model_cls"]:
                self.to_cuda()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cuda()
                self.transformer_weights.non_block_weights_to_cuda()

        if self.config["enable_cfg"]:
            if self.config["cfg_parallel"]:
                # ==================== CFG Parallel Processing ====================
                cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
                assert dist.get_world_size(cfg_p_group) == 2, "cfg_p_world_size must be equal to 2"
                cfg_p_rank = dist.get_rank(cfg_p_group)

                if cfg_p_rank == 0:
                    noise_pred = self._infer_cond_uncond(inputs, infer_condition=True)
                else:
                    noise_pred = self._infer_cond_uncond(inputs, infer_condition=False)

                noise_pred_list = [torch.zeros_like(noise_pred) for _ in range(2)]
                dist.all_gather(noise_pred_list, noise_pred, group=cfg_p_group)
                noise_pred_cond = noise_pred_list[0]  # cfg_p_rank == 0
                noise_pred_uncond = noise_pred_list[1]  # cfg_p_rank == 1
            else:
                # ==================== CFG Processing ====================
                noise_pred_cond = self._infer_cond_uncond(inputs, infer_condition=True)
                noise_pred_uncond = self._infer_cond_uncond(inputs, infer_condition=False)

            noise_pred_guided = noise_pred_uncond + self.scheduler.sample_guide_scale * (noise_pred_cond - noise_pred_uncond)
            self.scheduler.noise_pred_cond = noise_pred_cond
            self.scheduler.noise_pred_uncond = noise_pred_uncond
            self.scheduler.noise_pred_guided = noise_pred_guided
            self.scheduler.noise_pred = noise_pred_guided
        else:
            # ==================== No CFG ====================
            noise_pred = self._infer_cond_uncond(inputs, infer_condition=True)
            self.scheduler.noise_pred_cond = noise_pred
            self.scheduler.noise_pred_uncond = None
            self.scheduler.noise_pred_guided = noise_pred
            self.scheduler.noise_pred = noise_pred

        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == self.scheduler.infer_steps - 1 and "wan2.2_moe" not in self.config["model_cls"]:
                self.to_cpu()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cpu()
                self.transformer_weights.non_block_weights_to_cpu()
