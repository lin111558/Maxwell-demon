# ============================================================
# 新增：在右侧图表内嵌了专为论文截图设计的参数信息面板 (包含 δ, ε, 步数等)
# ============================================================
import os
import warnings

warnings.filterwarnings("ignore", message="Glyph .* missing from current font.*")

import matplotlib
from matplotlib import font_manager


def 设置中文字体_强制加载():
    """直接加载 Windows 字体文件，保证中文一定能显示（不打印提示）。"""
    字体路径候选 = [
        r"C:\Windows\Fonts\msyh.ttc",  # 微软雅黑
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",  # 黑体
        r"C:\Windows\Fonts\simsun.ttc",  # 宋体
    ]
    for path in 字体路径候选:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            font_name = font_manager.FontProperties(fname=path).get_name()
            matplotlib.rcParams["font.family"] = font_name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return


设置中文字体_强制加载()

# ============================================================
# 2) 常规 import
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from matplotlib.widgets import Button, Slider
from dataclasses import dataclass
from typing import List, Tuple


# ============================================================
# 3) 数学与模型工具
# ============================================================
def angle_lerp_deg(a: float, b: float, t: float) -> float:
    """按最短角差做角度插值（度），用于指针平滑追踪。"""
    a %= 360.0
    b %= 360.0
    diff = (b - a + 540.0) % 360.0 - 180.0
    return (a + t * diff) % 360.0


def build_generator_R(epsilon: float) -> np.ndarray:
    """生成 6x6 速率矩阵 R：R_ij = rate(to i | from j)（列为 from）"""
    if not (-1.0 < epsilon < 1.0):
        raise ValueError("epsilon 必须满足 |epsilon| < 1")

    R = np.zeros((6, 6))

    def add_bidirectional(i: int, j: int):
        R[i, j] = 1.0
        R[j, i] = 1.0

    add_bidirectional(0, 1)
    add_bidirectional(1, 2)
    add_bidirectional(3, 4)
    add_bidirectional(4, 5)

    # 耦合边 C0 <-> A1
    R[3, 2] = 1.0 - epsilon  # C0 -> A1
    R[2, 3] = 1.0 + epsilon  # A1 -> C0

    for j in range(6):
        R[j, j] = -np.sum(R[:, j]) + R[j, j]

    return R


def gillespie_path(R: np.ndarray, s0: int, tau: float, rng: np.random.Generator) -> Tuple[List[float], List[int]]:
    """Gillespie 算法：生成 [0,tau] 内的随机跳转轨迹"""
    t = 0.0
    s = int(s0)
    times = [0.0]
    states = [s]

    while t < tau:
        rates = R[:, s].copy()
        rates[s] = 0.0
        lam = float(rates.sum())
        if lam <= 0:
            break

        dt = rng.exponential(1.0 / lam)
        if t + dt > tau:
            break

        s = int(rng.choice(6, p=rates / lam))
        t += dt
        times.append(t)
        states.append(s)

    return times, states


def state_to_bit_pos(s: int) -> Tuple[int, int]:
    """6态 -> (bit层 0/1, 位置 0=A 1=B 2=C)"""
    return (0 if s < 3 else 1), (s % 3)


# ============================================================
# 4) 参数
# ============================================================
@dataclass
class Params:
    epsilon: float = 0.25
    tau: float = 1.0
    delta: float = 0.0

    tape_cells: int = 14
    read_idx: int = 5

    play_seconds: float = 0.9
    fps: int = 30

    follow_strength: float = 0.38
    radius_strength: float = 0.45

    window: int = 30
    phi_ylim: Tuple[float, float] = (-6, 6)

    use_blit: bool = False
    fast_steps: int = 200


# ============================================================
# 5) 主动画类
# ============================================================
class KSHAnimation:
    def __init__(self, P: Params):
        self.P = P
        self.rng = np.random.default_rng(seed=None)

        self.R = build_generator_R(P.epsilon)

        # 初始化两条带子，内容完全相同
        initial_bits = [int(self.rng.integers(0, 2)) for _ in range(P.tape_cells)]
        self.tape_before = initial_bits.copy()
        self.tape_after = initial_bits.copy()

        # 控制
        self.auto = False
        self.is_playing = False
        self.view_mode = "full"  # "full" 为完整历史，"window" 为滑动窗口

        # 记录 Φ 与累计 Φ
        self.step_count = 0
        self.phi_hist: List[float] = []
        self.phi_cum_hist: List[float] = []

        self.fig = plt.figure(figsize=(12.8, 5.2))
        self.ax_anim = self.fig.add_axes([0.05, 0.12, 0.58, 0.83])
        self.ax_plot = self.fig.add_axes([0.68, 0.20, 0.30, 0.70])

        self.ax_anim.set_xlim(0, 10)
        self.ax_anim.set_ylim(0, 6)
        self.ax_anim.axis("off")

        self._init_scene_plot()
        self._init_scene_anim()

        self._draw_tape(offset=0.0)

        interval_ms = max(1, int(1000 / max(1, P.fps)))
        self.timer = self.fig.canvas.new_timer(interval=interval_ms)
        self.timer.add_callback(self._on_timer)

    def _entering_idx(self) -> int:
        """进入高亮区的格子索引"""
        return (self.P.read_idx - 1) % self.P.tape_cells

    def _on_delta_change(self, val):
        self.P.delta = val
        self._refresh_info()
        self.fig.canvas.draw_idle()

    def _init_scene_anim(self):
        ax = self.ax_anim
        self.cx, self.cy = 3.2, 3.5
        self.r0, self.r1 = 1.0, 1.55
        ax.add_patch(Circle((self.cx, self.cy), self.r0, fill=False, lw=2))
        ax.add_patch(Circle((self.cx, self.cy), self.r1, fill=False, lw=2, alpha=0.8))

        self.angles = [210.0, 90.0, 330.0]
        for a, label in zip(self.angles, ["A", "B", "C"]):
            for r in [self.r0, self.r1]:
                x = self.cx + r * np.cos(np.deg2rad(a))
                y = self.cy + r * np.sin(np.deg2rad(a))
                ax.add_patch(Circle((x, y), 0.07, color="black"))
            x = self.cx + (self.r1 + 0.30) * np.cos(np.deg2rad(a))
            y = self.cy + (self.r1 + 0.30) * np.sin(np.deg2rad(a))
            ax.text(x, y, label, fontsize=14, ha="center", va="center", weight="bold")

        ax.text(self.cx, self.cy - self.r0 - 0.35, "bit=0", ha="center", fontsize=11)
        ax.text(self.cx, self.cy - self.r1 - 0.35, "bit=1", ha="center", fontsize=11, alpha=0.85)

        self.pointer_angle = 90.0
        self.pointer_radius = self.r0
        self.pointer = FancyArrowPatch((self.cx, self.cy), (self.cx, self.cy + 1),
                                       arrowstyle="-|>", lw=3, mutation_scale=18)
        ax.add_patch(self.pointer)
        self._update_pointer()

        self.cell_w = 0.5
        self.tape_x0 = 0.8
        self.tape_y_before = 1.0
        self.tape_y_after = 0.3

        ax.text(self.tape_x0 - 0.45, self.tape_y_before + 0.275, "作用前", fontsize=11, ha="right", va="center",
                weight="bold")
        ax.text(self.tape_x0 - 0.45, self.tape_y_after + 0.275, "作用后", fontsize=11, ha="right", va="center",
                weight="bold")

        self.cells_before, self.cells_after = [], []
        self.texts_before, self.texts_after = [], []

        for i in range(self.P.tape_cells):
            x = self.tape_x0 + i * self.cell_w
            rb = Rectangle((x, self.tape_y_before), self.cell_w, 0.55, fill=False, lw=1.5)
            ra = Rectangle((x, self.tape_y_after), self.cell_w, 0.55, fill=False, lw=1.5)
            ax.add_patch(rb);
            ax.add_patch(ra)
            self.cells_before.append(rb);
            self.cells_after.append(ra)

            tb = ax.text(x + self.cell_w / 2, self.tape_y_before + 0.275, "0", ha="center", va="center", fontsize=12)
            ta = ax.text(x + self.cell_w / 2, self.tape_y_after + 0.275, "0", ha="center", va="center", fontsize=12)
            self.texts_before.append(tb);
            self.texts_after.append(ta)

        rx = self.tape_x0 + self.P.read_idx * self.cell_w
        ax.add_patch(Rectangle((rx, self.tape_y_before), self.cell_w, 0.55, fill=True, alpha=0.18))
        ax.add_patch(Rectangle((rx, self.tape_y_after), self.cell_w, 0.55, fill=True, alpha=0.18))

        self.enter_marker = ax.text(0, 0, "▼", fontsize=14, color="#1f77b4", ha="center", va="bottom")
        self.enter_marker.set_visible(False)

        self.info = ax.text(6.2, 1.8, "", fontsize=10.5, linespacing=1.4)

        ax_step = self.fig.add_axes([0.16, 0.03, 0.09, 0.06])
        ax_toggle = self.fig.add_axes([0.26, 0.03, 0.10, 0.06])
        ax_auto = self.fig.add_axes([0.37, 0.03, 0.09, 0.06])
        ax_fast = self.fig.add_axes([0.47, 0.03, 0.09, 0.06])
        ax_view = self.fig.add_axes([0.57, 0.03, 0.09, 0.06])

        self.btn_step = Button(ax_step, "STEP")
        self.btn_toggle = Button(ax_toggle, "切换输入")
        self.btn_auto = Button(ax_auto, "Auto: OFF")
        self.btn_fast = Button(ax_fast, f"加速×{self.P.fast_steps}")
        self.btn_view = Button(ax_view, "视图: 全部")

        self.btn_step.on_clicked(self.step)
        self.btn_toggle.on_clicked(self.toggle_input)
        self.btn_auto.on_clicked(self.toggle_auto)
        self.btn_fast.on_clicked(self.fast_run)
        self.btn_view.on_clicked(self.toggle_view)

        ax_delta = self.fig.add_axes([0.16, 0.00, 0.50, 0.02])
        self.slider_delta = Slider(ax=ax_delta, label='δ', valmin=-0.49, valmax=0.49, valinit=self.P.delta,
                                   valstep=0.01, color="#ff7f0e")
        self.slider_delta.on_changed(self._on_delta_change)

        self._refresh_info("准备就绪：比特带上下分别为作用前后的对比。")

    def _init_scene_plot(self):
        ax = self.ax_plot
        ax.set_title("完整历史：信息流 Φ")
        ax.set_xlabel("interval 序号 n")
        ax.set_ylabel("Φ")
        self.current_ylim = self.P.phi_ylim
        ax.set_ylim(*self.current_ylim)
        self.line_phi, = ax.plot([], [], marker="o", linewidth=0.8, markersize=4, label="Φ(n)")
        self.line_phi_cum, = ax.plot([], [], linewidth=3.2, alpha=0.6, label="累计 Φ")

        # 将图例移到左上角，给右上角留出空间
        ax.legend(loc="upper left")
        ax.set_xlim(1, self.P.window + 1)
        ax.set_autoscaley_on(False)

        # 新增：在图表内部右上角添加一个文本框，作为论文截图专用的信息面板
        # 使用 axes 坐标系 (0~1)，0.96表示靠右，0.95表示靠上
        self.plot_param_text = ax.text(
            0.96, 0.95, "", transform=ax.transAxes,
            ha="right", va="top", fontsize=10.5, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.85)
        )

    def _draw_tape(self, offset: float):
        for i in range(self.P.tape_cells):
            x = self.tape_x0 + i * self.cell_w + offset
            self.texts_before[i].set_position((x + self.cell_w / 2, self.tape_y_before + 0.275))
            self.texts_before[i].set_text(str(self.tape_before[i]))
            self.texts_after[i].set_position((x + self.cell_w / 2, self.tape_y_after + 0.275))
            self.texts_after[i].set_text(str(self.tape_after[i]))

    def _update_enter_marker(self, offset: float):
        idx = self._entering_idx()
        x = self.tape_x0 + idx * self.cell_w + offset + self.cell_w / 2
        self.enter_marker.set_position((x, self.tape_y_before + 0.62))
        self.enter_marker.set_visible(True)

    def _update_pointer(self):
        x = self.cx + self.pointer_radius * np.cos(np.deg2rad(self.pointer_angle))
        y = self.cy + self.pointer_radius * np.sin(np.deg2rad(self.pointer_angle))
        self.pointer.set_positions((self.cx, self.cy), (x, y))

    def _refresh_info(self, extra: str = ""):
        entering_bit = int(self.tape_before[self._entering_idx()])
        phi_cum = self.phi_cum_hist[-1] if self.phi_cum_hist else 0.0

        # 1. 更新左侧原有的全局信息区域
        self.info.set_text(
            f"步数 n      : {self.step_count}\n"
            f"ε           : {self.P.epsilon:+.3f}\n"
            f"δ           : {self.P.delta:+.2f}\n"
            f"输入bit  : {entering_bit}\n"
            f"累计 Φ      : {phi_cum:+.0f}\n"
            f"{extra}"
        )

        # 2. 更新右侧曲线图内的专属信息面板（为截图优化）
        plot_info_str = (
            f"步数 n : {self.step_count}\n"
            f"累计 Φ : {phi_cum:+.0f}\n"
            f"负载 ε : {self.P.epsilon:+.3f}\n"
            f"超额 δ : {self.P.delta:+.2f}"
        )
        self.plot_param_text.set_text(plot_info_str)

    def toggle_view(self, event=None):
        if self.view_mode == "full":
            self.view_mode = "window"
            self.btn_view.label.set_text("视图: 30步")
            self.ax_plot.set_title("最近30步：信息流 Φ")
        else:
            self.view_mode = "full"
            self.btn_view.label.set_text("视图: 全部")
            self.ax_plot.set_title("完整历史：信息流 Φ")
        self._update_curves()

    def _update_curves(self):
        N = len(self.phi_hist)
        if N == 0: return
        win = int(self.P.window)

        if self.view_mode == "window":
            start = max(0, N - win)
        else:
            start = 0

        n = np.arange(start + 1, N + 1)
        phi = np.array(self.phi_hist[start:], dtype=float)
        phi_cum = np.array(self.phi_cum_hist[start:], dtype=float)
        self.line_phi.set_data(n, phi)
        self.line_phi_cum.set_data(n, phi_cum)

        all_vals = np.concatenate([phi, phi_cum])
        val_min, val_max = np.nanmin(all_vals), np.nanmax(all_vals)
        max_abs = max(abs(val_min), abs(val_max))
        y_bound = max(((max_abs + 9) // 10) * 10, self.P.phi_ylim[1])
        if (-y_bound, y_bound) != self.current_ylim:
            self.ax_plot.set_ylim(-y_bound, y_bound)
            self.current_ylim = (-y_bound, y_bound)

        if self.view_mode == "window":
            self.ax_plot.set_xlim(max(1, N - win + 1), max(win + 1, N + 1))
        else:
            self.ax_plot.set_xlim(1, max(N + 1, win + 1))

        self.fig.canvas.draw_idle()

    def toggle_input(self, event=None):
        if self.is_playing: return
        idx = self._entering_idx()
        new_val = 1 - int(self.tape_before[idx])
        self.tape_before[idx] = new_val
        self.tape_after[idx] = new_val
        self._draw_tape(0)
        self._update_enter_marker(0)
        self.fig.canvas.draw_idle()

    def toggle_auto(self, event=None):
        self.auto = not self.auto
        self.btn_auto.label.set_text("Auto: ON" if self.auto else "Auto: OFF")
        if self.auto and (not self.is_playing): self.step()

    def step(self, event=None):
        if self.is_playing: return
        self.enter_idx = self._entering_idx()
        self.in_bit = int(self.tape_before[self.enter_idx])
        self.enter_marker.set_visible(True)
        self._update_enter_marker(0.0)

        pos0 = int(self.rng.integers(0, 3))
        s0 = pos0 + (3 if self.in_bit == 1 else 0)
        self.times, self.states = gillespie_path(self.R, s0, self.P.tau, self.rng)

        self.frame, self.total_frames = 0, max(2, int(self.P.play_seconds * self.P.fps))
        self.is_playing = True
        self.timer.start()

    def _on_timer(self):
        u = self.frame / (self.total_frames - 1)
        local_t = u * self.P.tau
        self._draw_tape(offset=u * self.cell_w)
        self._update_enter_marker(offset=u * self.cell_w)

        idx = np.searchsorted(self.times, local_t, side="right") - 1
        s_now = int(self.states[int(np.clip(idx, 0, len(self.states) - 1))])
        bit_now, pos_now = state_to_bit_pos(s_now)
        self.pointer_angle = angle_lerp_deg(self.pointer_angle, self.angles[pos_now], self.P.follow_strength)
        self.pointer_radius += self.P.radius_strength * ((self.r1 if bit_now == 1 else self.r0) - self.pointer_radius)
        self._update_pointer()

        self.frame += 1
        if self.frame >= self.total_frames:
            self.timer.stop();
            self.is_playing = False
            out_bit, _ = state_to_bit_pos(int(self.states[-1]))

            p1 = 0.5 + self.P.delta
            new_bit = 1 if self.rng.random() < p1 else 0

            self.tape_before.pop(-1);
            self.tape_before.insert(0, new_bit)
            self.tape_after.pop(-1);
            self.tape_after.insert(0, new_bit)

            self.tape_after[self.P.read_idx] = int(out_bit)

            phi_i = float(out_bit - self.in_bit)
            self.phi_hist.append(phi_i)
            self.phi_cum_hist.append((self.phi_cum_hist[-1] if self.phi_cum_hist else 0.0) + phi_i)
            self.step_count += 1

            self._draw_tape(0.0);
            self.enter_marker.set_visible(False)
            self._refresh_info(f"结束：输出bit={out_bit} | Φ={phi_i:+.0f}")
            self._update_curves()
            if self.auto: self.step()

    def fast_run(self, event=None):
        if self.is_playing: return
        for _ in range(self.P.fast_steps):
            in_bit = int(self.tape_before[self._entering_idx()])
            s0 = int(self.rng.integers(0, 3)) + (3 if in_bit == 1 else 0)
            _, states = gillespie_path(self.R, s0, self.P.tau, self.rng)
            out_bit, _ = state_to_bit_pos(int(states[-1]))

            new_bit = 1 if self.rng.random() < (0.5 + self.P.delta) else 0
            self.tape_before.pop(-1);
            self.tape_before.insert(0, new_bit)
            self.tape_after.pop(-1);
            self.tape_after.insert(0, new_bit)
            self.tape_after[self.P.read_idx] = int(out_bit)

            phi_i = float(out_bit - in_bit)
            self.phi_hist.append(phi_i)
            self.phi_cum_hist.append((self.phi_cum_hist[-1] if self.phi_cum_hist else 0.0) + phi_i)
            self.step_count += 1

        self._draw_tape(0.0);
        self._update_curves()
        self._refresh_info(f"加速完成：共执行 {self.P.fast_steps} 步。")

    def show(self):
        plt.show()


if __name__ == "__main__":
    P = Params(epsilon=0.4, tau=1.0, delta=0.0)
    KSHAnimation(P).show()