import gc

import torch
from loguru import logger

from lightx2v.common.flowcache import SFFlowCacheManager
from lightx2v.common.kvcache import KVCacheManager
from lightx2v.models.networks.wan.sf_model import WanSFModel
from lightx2v.models.runners.wan.wan_runner import WanRunner, build_wan_model_with_lora
from lightx2v.models.schedulers.wan.self_forcing.scheduler import WanSFScheduler
from lightx2v.models.video_encoders.hf.wan.vae_sf import WanSFVAE
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.async_vae import AsyncVAEChunkDecoder
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import get_rank_and_world_size, wan_vae_to_comfy
from lightx2v.utils.video_recorder import VideoRecorder


@RUNNER_REGISTER("wan2.1_sf")
class WanSFRunner(WanRunner):
    def __init__(self, config):
        super().__init__(config)
        self.is_live = config.get("is_live", False)
        if self.is_live:
            self.vae_cls = WanSFVAE
            self.width = self.config["target_width"]
            self.height = self.config["target_height"]
            self.run_main = self.run_main_live

    def load_transformer(self):
        wan_model_kwargs = {"model_path": self.config["model_path"], "config": self.config, "device": self.init_device}
        lora_configs = self.config.get("lora_configs")
        if not lora_configs:
            model = WanSFModel(**wan_model_kwargs)
        else:
            model = build_wan_model_with_lora(WanSFModel, self.config, wan_model_kwargs, lora_configs, model_type="wan2.1")
        return model

    def init_scheduler(self):
        self.scheduler = WanSFScheduler(self.config)

    def init_kv_cache_manager(self):
        self.model.kv_cache_manager = KVCacheManager(config=self.config, device=torch.device("cuda"), sp_group=self.model.seq_p_group)
        self.model.kv_cache_manager._create_kv_caches(self.input_info.latent_shape)
        self.model.transformer_infer.kv_cache_manager = self.model.kv_cache_manager
        self.input_info.latent_shape = [self.input_info.latent_shape[0], self.model.kv_cache_manager.num_output_frames, self.input_info.latent_shape[2], self.input_info.latent_shape[3]]
        self.scheduler.num_output_frames = self.model.kv_cache_manager.num_output_frames
        self.scheduler.num_chunks = self.model.kv_cache_manager.num_output_frames // self.config.get("ar_config", {}).get("num_frame_per_chunk", 3)
        self.flowcache_manager = SFFlowCacheManager(self.config)
        self.flowcache_manager.reset()
        self.model.flowcache_manager = self.flowcache_manager
        self.model.transformer_infer.flowcache_manager = self.flowcache_manager

    def _run_sf_step(self, segment_idx: int, step_index: int, *, is_rerun: bool) -> None:
        fc = getattr(self, "flowcache_manager", None)
        if fc is not None and fc.uses_feature_cache:
            metric = fc.compute_metric(self.model, self.inputs)
            if fc.should_skip_transformer(segment_idx, step_index, metric, is_rerun=is_rerun):
                fc.apply_cached_noise_pred(self.model.scheduler, segment_idx)
            else:
                self.model.infer(self.inputs)
                if not is_rerun:
                    seg_start = segment_idx * self.scheduler.num_frame_per_chunk
                    seg_end = min((segment_idx + 1) * self.scheduler.num_frame_per_chunk, self.scheduler.num_output_frames)
                    noise_pred = self.model.scheduler.noise_pred[:, seg_start:seg_end]
                    fc.on_forward(segment_idx, step_index, metric, noise_pred)
        else:
            self.model.infer(self.inputs)

    def get_video_segment_num(self):
        self.video_segment_num = self.scheduler.num_chunks

    @ProfilingContext4DebugL1("Run VAE Decoder")
    def run_vae_decoder(self, latents):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae_decoder = self.load_vae_decoder()
        if self.is_live:
            images = self.vae_decoder.decode(latents.to(GET_DTYPE()), use_cache=True)
        else:
            images = self.vae_decoder.decode(latents.to(GET_DTYPE()))
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae_decoder
            torch.cuda.empty_cache()
            gc.collect()
        return images

    def init_run(self):
        if (self.config.get("lazy_load", False) or self.config.get("unload_modules", False)) and not getattr(self, "model", None):
            self.model = self.load_transformer()
            self.model.set_scheduler(self.scheduler)
        self.init_kv_cache_manager()
        super().init_run()

    def end_run(self):
        self.model.kv_cache_manager.save_calibration()
        fc = getattr(self, "flowcache_manager", None)
        if fc is not None:
            fc.log_stats()
        super().end_run()

    def run_segment(self, segment_idx=0):
        infer_steps = self.model.scheduler.infer_steps
        fc = getattr(self, "flowcache_manager", None)
        if fc is not None:
            fc.begin_chunk(segment_idx)
        for step_index in range(infer_steps):
            # only for single segment, check stop signal every step
            if self.video_segment_num == 1:
                self.check_stop()
            logger.info(f"==> step_index: {step_index + 1} / {infer_steps}")

            self.model.kv_cache_manager.current_step = step_index

            with ProfilingContext4DebugL1("step_pre"):
                self.model.scheduler.step_pre(seg_index=segment_idx, step_index=step_index, is_rerun=False)

            with ProfilingContext4DebugL1("🚀 infer_main"):
                self._run_sf_step(segment_idx, step_index, is_rerun=False)

            with ProfilingContext4DebugL1("step_post"):
                self.model.scheduler.step_post()

            if self.progress_callback:
                current_step = segment_idx * infer_steps + step_index + 1
                total_all_steps = self.video_segment_num * infer_steps
                self.progress_callback((current_step / total_all_steps) * 100, 100)

        return self.model.scheduler.stream_output

    def decode_segment_latents(self, segment_idx: int, latents: torch.Tensor) -> torch.Tensor:
        return self.run_vae_decoder(latents.detach().clone())

    def init_video_recorder(self):
        output_video_path = self.input_info.save_result_path
        self.video_recorder = None
        if isinstance(output_video_path, dict):
            output_video_path = output_video_path["data"]
        logger.info(f"init video_recorder with output_video_path: {output_video_path}")
        rank, world_size = get_rank_and_world_size()
        if output_video_path and rank == world_size - 1:
            record_fps = self.config.get("target_fps", 16)
            if "video_frame_interpolation" in self.config and self.vfi_model is not None:
                record_fps = self.config["video_frame_interpolation"]["target_fps"]

            self.video_recorder = VideoRecorder(
                livestream_url=output_video_path,
                fps=record_fps,
            )

    @ProfilingContext4DebugL1("End run segment")
    def end_run_segment(self, segment_idx=None):
        with ProfilingContext4DebugL1("step_pre_in_rerun"):
            self.model.scheduler.step_pre(
                seg_index=segment_idx,
                step_index=self.model.scheduler.infer_steps - 1,
                is_rerun=True,
            )
        with ProfilingContext4DebugL1("🚀 infer_main_in_rerun"):
            self._run_sf_step(segment_idx, self.model.scheduler.infer_steps - 1, is_rerun=True)

        fc = getattr(self, "flowcache_manager", None)
        if fc is not None:
            fc.mark_chunk_completed(segment_idx)
            fc.maybe_compress_kv(self.model, segment_idx)

        self.gen_video_final = torch.cat([self.gen_video_final, self.gen_video], dim=0) if self.gen_video_final is not None else self.gen_video
        if self.is_live:
            if self.video_recorder:
                stream_video = wan_vae_to_comfy(self.gen_video)
                self.video_recorder.pub_video(stream_video)

        torch.cuda.empty_cache()

    @ProfilingContext4DebugL2("Run DiT")
    def run_main(self, total_steps=None):
        self.init_run()
        if self.config.get("compile", False):
            self.model.select_graph_for_compile(self.input_info)

        lazy_vae = self.config.get("lazy_load", False) or self.config.get("unload_modules", False)
        if lazy_vae:
            self.vae_decoder = self.load_vae_decoder()
        vae_decoder = AsyncVAEChunkDecoder.from_config(self.config, device=torch.device("cuda"), vae_decoder=self.vae_decoder)

        with (
            no_sync_profiling(enabled=vae_decoder.is_async),
            ProfilingContext4DebugL1(
                f"AR chunk total {self.video_segment_num} chunks",
                recorder_mode=GET_RECORDER_MODE(),
                metrics_func=monitor_cli.lightx2v_run_segments_end2end_duration,
                metrics_labels=["DefaultRunner"],
            ),
        ):
            try:
                for segment_idx in range(self.video_segment_num):
                    logger.info(f"start chunk {segment_idx + 1}/{self.video_segment_num}")
                    with ProfilingContext4DebugL1(
                        f"chunk end2end {segment_idx + 1}/{self.video_segment_num}",
                        recorder_mode=GET_RECORDER_MODE(),
                        metrics_func=monitor_cli.lightx2v_run_segments_end2end_duration,
                        metrics_labels=["DefaultRunner"],
                    ):
                        self.check_stop()
                        self.init_run_segment(segment_idx)
                        latents = self.run_segment(segment_idx)

                        with ProfilingContext4DebugL1("step_pre_in_rerun"):
                            self.model.scheduler.step_pre(
                                seg_index=segment_idx,
                                step_index=self.model.scheduler.infer_steps - 1,
                                is_rerun=True,
                            )
                        with ProfilingContext4DebugL1("infer_main_in_rerun"):
                            self._run_sf_step(segment_idx, self.model.scheduler.infer_steps - 1, is_rerun=True)

                        fc = getattr(self, "flowcache_manager", None)
                        if fc is not None:
                            fc.mark_chunk_completed(segment_idx)
                            fc.maybe_compress_kv(self.model, segment_idx)

                    vae_decoder.submit(self.decode_segment_latents, segment_idx, latents)
                    torch.cuda.empty_cache()
                decoded_chunks = vae_decoder.finish()
            finally:
                if "vae_decoder" in locals():
                    vae_decoder.finish()
                if lazy_vae:
                    del self.vae_decoder
                    torch.cuda.empty_cache()
                    gc.collect()

        self.gen_video = torch.cat(decoded_chunks, dim=0)
        self.gen_video_final = self.gen_video
        gen_video_final = self.process_images_after_vae_decoder()
        self.end_run()
        return gen_video_final

    @ProfilingContext4DebugL2("Run DiT")
    def run_main_live(self, total_steps=None):
        try:
            self.init_video_recorder()
            logger.info(f"init video_recorder: {self.video_recorder}")
            rank, world_size = get_rank_and_world_size()
            if rank == world_size - 1:
                assert self.video_recorder is not None, "video_recorder is required for stream audio input for rank 2"
                self.video_recorder.start(self.width, self.height)
            if world_size > 1:
                dist.barrier()
            self.init_run()
            if self.config.get("compile", False):
                self.model.select_graph_for_compile(self.input_info)

            for segment_idx in range(self.video_segment_num):
                logger.info(f"🔄 start segment {segment_idx + 1}/{self.video_segment_num}")
                with ProfilingContext4DebugL1(
                    f"segment end2end {segment_idx + 1}/{self.video_segment_num}",
                    recorder_mode=GET_RECORDER_MODE(),
                    metrics_func=monitor_cli.lightx2v_run_segments_end2end_duration,
                    metrics_labels=["DefaultRunner"],
                ):
                    self.check_stop()
                    # 1. default do nothing
                    self.init_run_segment(segment_idx)
                    # 2. main inference loop
                    latents = self.run_segment(segment_idx)
                    # 3. vae decoder
                    self.gen_video = self.run_vae_decoder(latents)
                    # 4. default do nothing
                    self.end_run_segment(segment_idx)
        finally:
            if hasattr(self.model, "inputs"):
                self.end_run()
            if self.video_recorder:
                self.video_recorder.stop()
                self.video_recorder = None
