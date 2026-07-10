"""基于 tkinter 的弹道模拟 GUI 主应用。"""

from __future__ import annotations

import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

from ballistic_sim.cli import _compute_summary
from ballistic_sim.config import MonteCarloConfig, SimConfig
from ballistic_sim.gui.builder import (
    build_config_and_phases,
    default_preset,
    dump_config_to_yaml,
    list_presets,
    load_config_from_yaml,
    submodel_classes,
)
from ballistic_sim.gui.fields import build_form, read_model_variables, update_model_variables
from ballistic_sim.gui.runner import SimulationRunner
from ballistic_sim.simulator import SimResult
from ballistic_sim.viz import attach_launch_lla
from ballistic_sim.viz.groundtrack import plot_groundtrack, plot_impact_summary
from ballistic_sim.viz.interactive3d import plot_trajectory_3d
from ballistic_sim.viz.profile import plot_altitude_range, plot_velocity_profile
from ballistic_sim.viz.trajectory3d import plot_trajectory3d, plot_trajectory3d_topdown


class BallisticGuiApp(ttk.Frame):
    """弹道模拟 GUI 主窗口内容面板。"""

    def __init__(self, master: tk.Tk, *, mission: str = "projectile") -> None:
        super().__init__(master)
        self.master = master
        self.master.title("Ballistic Sim GUI")
        self.master.geometry("1200x800")

        self._runner = SimulationRunner()
        self._poll_id: Optional[str] = None
        self._last_result: Optional[SimResult] = None

        self._build_toolbar()
        self._build_paned_body()

        self._current_mission = mission
        self._current_preset: Optional[str] = default_preset(mission)
        self._mission_var.set(mission)
        self._preset_var.set(self._current_preset or "")
        self._update_preset_list()
        self._load_config_to_forms()

    # -------------------------------------------------------------------------
    # 界面构造
    # -------------------------------------------------------------------------
    def _build_toolbar(self) -> None:
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=4, pady=4)

        ttk.Label(toolbar, text="任务:").pack(side="left", padx=(0, 4))
        self._mission_var = tk.StringVar(value="projectile")
        mission_combo = ttk.Combobox(
            toolbar,
            textvariable=self._mission_var,
            values=["projectile", "missile", "rocket", "icbm", "suborbital"],
            state="readonly",
            width=12,
        )
        mission_combo.pack(side="left", padx=(0, 12))
        mission_combo.bind("<<ComboboxSelected>>", self._on_mission_changed)

        ttk.Label(toolbar, text="预设:").pack(side="left", padx=(0, 4))
        self._preset_var = tk.StringVar()
        self._preset_combo = ttk.Combobox(
            toolbar, textvariable=self._preset_var, state="readonly", width=14
        )
        self._preset_combo.pack(side="left", padx=(0, 12))
        self._preset_combo.bind("<<ComboboxSelected>>", self._on_preset_changed)

        ttk.Button(toolbar, text="Run", command=self._on_run).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Load YAML", command=self._on_load_yaml).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Save YAML", command=self._on_save_yaml).pack(side="left", padx=4)

    def _build_paned_body(self) -> None:
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=4, pady=4)

        self._notebook = ttk.Notebook(paned)
        self._form_frames: Dict[str, ttk.Frame] = {}
        self._form_vars: Dict[str, Dict[str, tk.Variable]] = {}
        for name in ["vehicle", "launch", "environment", "guidance", "options"]:
            frame = ttk.Frame(self._notebook)
            inner = ttk.Frame(frame)
            inner.pack(fill="both", expand=True, padx=4, pady=4)
            self._notebook.add(frame, text=name.capitalize())
            self._form_frames[name] = inner

        # Monte Carlo 是 OptionsConfig 的子模型，单独一页
        mc_frame = ttk.Frame(self._notebook)
        mc_inner = ttk.Frame(mc_frame)
        mc_inner.pack(fill="both", expand=True, padx=4, pady=4)
        self._notebook.add(mc_frame, text="Monte Carlo")
        self._form_frames["monte_carlo"] = mc_inner

        paned.add(self._notebook, weight=1)

        right = ttk.Frame(paned)
        paned.add(right, weight=2)

        self._summary_text = tk.Text(right, wrap="word", height=10)
        self._summary_text.pack(fill="x", padx=4, pady=4)

        self._right_notebook = ttk.Notebook(right)
        self._right_notebook.pack(fill="both", expand=True, padx=4, pady=4)

        import matplotlib

        matplotlib.use("TkAgg")

        self._plot_tabs: List[Tuple[str, ttk.Frame, Optional[Callable[[SimResult], Any]]]] = []

        def _add_plot_tab(label: str, plot_fn: Optional[Callable[[SimResult], Any]]) -> ttk.Frame:
            frame = ttk.Frame(self._right_notebook)
            self._right_notebook.add(frame, text=label)
            self._plot_tabs.append((label, frame, plot_fn))
            return frame

        _add_plot_tab("高度-射程", plot_altitude_range)
        _add_plot_tab("高度/速度-时间", plot_velocity_profile)
        _add_plot_tab("地面航迹", plot_groundtrack)
        _add_plot_tab("落点摘要", plot_impact_summary)
        _add_plot_tab("3D 弹道", plot_trajectory3d)
        _add_plot_tab("顶视/侧视", plot_trajectory3d_topdown)

        tab_3d = _add_plot_tab("3D 轨迹 (plotly)", None)
        self._3d_status_label = tk.Label(
            tab_3d,
            text="点击按钮生成 3D 轨迹（需安装 plotly）",
            wraplength=600,
        )
        self._3d_status_label.pack(pady=12)
        ttk.Button(
            tab_3d,
            text="生成 3D 轨迹",
            command=self._on_generate_3d,
        ).pack(pady=4)

        tab_events = _add_plot_tab("事件日志", None)
        self._event_tree = ttk.Treeview(
            tab_events,
            columns=("phase", "name", "t"),
            show="headings",
        )
        self._event_tree.heading("phase", text="阶段")
        self._event_tree.heading("name", text="事件")
        self._event_tree.heading("t", text="时刻 (s)")
        self._event_tree.column("phase", width=120)
        self._event_tree.column("name", width=180)
        self._event_tree.column("t", width=120)
        self._event_tree.pack(fill="both", expand=True, padx=4, pady=4)

    # -------------------------------------------------------------------------
    # 事件处理
    # -------------------------------------------------------------------------
    def _on_mission_changed(self, _event: Any = None) -> None:
        mission = self._mission_var.get()
        self._current_mission = mission
        self._current_preset = default_preset(mission)
        self._update_preset_list()
        self._preset_var.set(self._current_preset or "")
        self._load_config_to_forms()

    def _on_preset_changed(self, _event: Any = None) -> None:
        self._current_preset = self._preset_var.get() or None
        self._load_config_to_forms()

    def _update_preset_list(self) -> None:
        presets = list_presets(self._current_mission)
        self._preset_combo["values"] = presets
        if presets and self._current_preset not in presets:
            self._current_preset = presets[0]
            self._preset_var.set(self._current_preset)

    # -------------------------------------------------------------------------
    # 配置读写
    # -------------------------------------------------------------------------
    def _load_config_to_forms(self) -> None:
        cfg, _ = build_config_and_phases(self._current_mission, self._current_preset)
        self._build_forms_if_needed(cfg)
        self._update_forms_from_config(cfg)

    def _build_forms_if_needed(self, cfg: SimConfig) -> None:
        classes = submodel_classes()
        for name, model_cls in classes.items():
            if name not in self._form_vars:
                self._form_vars[name] = build_form(self._form_frames[name], model_cls)

        mc_cfg = cfg.options.monte_carlo or MonteCarloConfig()
        if "monte_carlo" not in self._form_vars:
            self._form_vars["monte_carlo"] = build_form(self._form_frames["monte_carlo"], mc_cfg)

    def _update_forms_from_config(self, cfg: SimConfig) -> None:
        for name, model in [
            ("vehicle", cfg.vehicle),
            ("launch", cfg.launch),
            ("environment", cfg.environment),
            ("guidance", cfg.guidance),
            ("options", cfg.options),
        ]:
            update_model_variables(self._form_vars[name], model)

        mc_cfg = cfg.options.monte_carlo or MonteCarloConfig()
        update_model_variables(self._form_vars["monte_carlo"], mc_cfg)

    def _read_config_from_forms(self) -> SimConfig:
        classes = submodel_classes()
        kwargs: Dict[str, Any] = {"mission": self._current_mission}
        for name, model_cls in classes.items():
            vars_ = self._form_vars[name]
            data = read_model_variables(model_cls, vars_)
            kwargs[name] = model_cls(**data)

        mc_vars = self._form_vars["monte_carlo"]
        mc_data = read_model_variables(MonteCarloConfig, mc_vars)
        kwargs["options"] = kwargs["options"].model_copy(
            update={"monte_carlo": MonteCarloConfig(**mc_data)}
        )

        return SimConfig(**kwargs)

    # -------------------------------------------------------------------------
    # 运行仿真
    # -------------------------------------------------------------------------
    def _on_run(self) -> None:
        if self._runner.is_alive():
            messagebox.showwarning("运行中", "仿真已在后台运行，请等待完成。")
            return

        try:
            cfg = self._read_config_from_forms()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("配置错误", f"读取参数失败: {exc}")
            return

        self._summary_text.delete("1.0", "end")
        self._summary_text.insert("end", "正在运行仿真...\n")
        self._runner.run(cfg)
        self._schedule_poll(cfg)

    def _schedule_poll(self, cfg: SimConfig) -> None:
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except Exception:  # noqa: BLE001
                pass

        def _poll() -> None:
            try:
                kind, payload = self._runner.result_queue.get_nowait()
            except queue.Empty:
                self._poll_id = self.after(100, _poll)
                return

            if kind == "ok":
                result = payload
                self._display_result(cfg, result)
            else:
                messagebox.showerror("仿真错误", f"运行失败: {payload}")
                self._summary_text.insert("end", f"错误: {payload}\n")
            self._poll_id = None

        self._poll_id = self.after(100, _poll)

    def _on_generate_3d(self) -> None:
        """生成 3D 轨迹 HTML 并显示保存路径（无显示环境时优雅降级）。"""
        if self._last_result is None or self._last_result.y.size == 0:
            self._3d_status_label.config(text="尚无可用仿真结果")
            return

        try:
            import plotly  # noqa: F401
        except ImportError:
            self._3d_status_label.config(
                text="plotly 未安装，请执行 pip install ballistic_sim[viz3d]"
            )
            return

        import tempfile
        import webbrowser

        html_path = Path(tempfile.gettempdir()) / "ballistic_sim_3d.html"
        try:
            plot_trajectory_3d(self._last_result, output_path=html_path)
            self._3d_status_label.config(text=f"3D 轨迹已保存: {html_path}")
            try:
                webbrowser.open(f"file://{html_path}")
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001
            self._3d_status_label.config(text=f"生成 3D 轨迹失败: {exc}")

    def _display_result(self, cfg: SimConfig, result: SimResult) -> None:
        self._last_result = result
        attach_launch_lla(result, cfg.launch.lat_deg, cfg.launch.lon_deg, cfg.launch.alt_m)
        summary = _compute_summary(cfg, result)

        self._summary_text.delete("1.0", "end")
        lines = [
            f"Mission: {cfg.mission}",
            f"Stop   : {summary.get('stop_reason')}",
            f"TOF    : {summary.get('t_end_s')} s",
        ]
        if "range_m" in summary:
            lines.append(f"Range  : {summary['range_m'] / 1e3:.2f} km")
        if "max_alt_m" in summary:
            lines.append(f"Max Alt: {summary['max_alt_m'] / 1e3:.2f} km")
        if "impact_speed_m_s" in summary:
            lines.append(f"Impact : {summary['impact_speed_m_s']:.1f} m/s")
        if "v_end_m_s" in summary:
            lines.append(f"V_end  : {summary['v_end_m_s']:.1f} m/s")
        if "orbit" in summary:
            orbit = summary["orbit"]
            lines.append(
                f"Orbit  : a={orbit['a_m']/1e3:.1f} km, e={orbit['e']:.3f}, "
                f"i={orbit['i_deg']:.1f}°, h_peri={orbit['h_peri_km']:.1f} km"
            )
        self._summary_text.insert("end", "\n".join(lines) + "\n")

        self._refresh_plot_tabs(result)
        self._refresh_event_tree(result)

    def _refresh_plot_tabs(self, result: SimResult) -> None:
        """根据最新仿真结果刷新所有 matplotlib 图表页签。"""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        for label, frame, plot_fn in self._plot_tabs:
            if plot_fn is None:
                continue
            for w in frame.winfo_children():
                w.destroy()
            try:
                fig = plot_fn(result)
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
                canvas.draw()
            except Exception as exc:  # noqa: BLE001
                msg = tk.Label(frame, text=f"[{label}] 绘图失败: {exc}", wraplength=600)
                msg.pack(expand=True)

    def _refresh_event_tree(self, result: SimResult) -> None:
        """刷新事件日志树。"""
        for item in self._event_tree.get_children():
            self._event_tree.delete(item)
        for ev in result.event_log:
            t = ev.get("t")
            t_str = f"{t:.3f}" if isinstance(t, (int, float)) else str(t)
            self._event_tree.insert(
                "",
                "end",
                values=(
                    ev.get("phase", ""),
                    ev.get("name", ""),
                    t_str,
                ),
            )

    # -------------------------------------------------------------------------
    # YAML 加载/保存
    # -------------------------------------------------------------------------
    def _on_load_yaml(self) -> None:
        path_str = filedialog.askopenfilename(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if not path_str:
            return
        try:
            cfg = load_config_from_yaml(Path(path_str))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("加载失败", f"无法加载 YAML: {exc}")
            return

        self._current_mission = cfg.mission
        self._mission_var.set(cfg.mission)
        self._current_preset = None
        self._update_preset_list()
        self._preset_var.set("")
        self._build_forms_if_needed(cfg)
        self._update_forms_from_config(cfg)

    def _on_save_yaml(self) -> None:
        path_str = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if not path_str:
            return
        try:
            cfg = self._read_config_from_forms()
            dump_config_to_yaml(cfg, Path(path_str))
            messagebox.showinfo("保存成功", f"配置已保存到 {path_str}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("保存失败", f"无法保存 YAML: {exc}")

    # -------------------------------------------------------------------------
    # 公共属性
    # -------------------------------------------------------------------------
    @property
    def current_mission(self) -> str:
        return self._current_mission

    @property
    def current_preset(self) -> Optional[str]:
        return self._current_preset
