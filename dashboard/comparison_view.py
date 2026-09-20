"""
Interactive Side-by-Side Pygame Comparison Dashboard.
Visualizes Fixed-Timer Baseline (Left) vs Trained Q-Learning Agent (Right) simultaneously
with synchronized vehicle spawning, multi-tier emergency vehicles (Ambulances & Fire Trucks),
interactive conflicting emergency arbitration triggers, and Start/Stop controls.
"""
import sys
import os
import pygame
from environment.intersection_env import (
    IntersectionEnv,
    PHASE_NS_GREEN,
    PHASE_EW_GREEN,
)
from environment.vehicle import (
    VEHICLE_TYPE_AMBULANCE,
    VEHICLE_TYPE_FIRETRUCK,
)
from environment.renderer import IntersectionRenderer
from controllers.fixed_timer_controller import FixedTimerController
from controllers.rl_agent import QLearningAgent
from training.train import train_agent


# Dashboard Color Palette
DARK_BG = (18, 22, 28)
PANEL_BG = (26, 31, 38)
PANEL_BORDER = (48, 54, 61)
TEXT_WHITE = (240, 246, 252)
TEXT_MUTED = (139, 148, 158)
ACCENT_BLUE = (88, 166, 255)
ACCENT_GREEN = (63, 185, 80)
ACCENT_ORANGE = (210, 153, 34)
ACCENT_RED = (248, 81, 73)
ACCENT_FIRE = (255, 75, 75)
ACCENT_AMB = (80, 190, 255)

# Button Colors
BTN_START_BG = (40, 120, 60)
BTN_START_HOVER = (50, 150, 75)
BTN_STOP_BG = (160, 50, 50)
BTN_STOP_HOVER = (200, 65, 65)
BTN_RESET_BG = (60, 80, 120)
BTN_RESET_HOVER = (75, 100, 150)


class Button:
    """Simple clickable button widget."""

    def __init__(self, x, y, w, h, text, bg_color, hover_color, text_color=TEXT_WHITE):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.hovered = False

    def draw(self, surface, font):
        color = self.hover_color if self.hovered else self.bg_color
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        pygame.draw.rect(surface, (255, 255, 255, 80), self.rect, width=1, border_radius=6)
        txt_surf = font.render(self.text, True, self.text_color)
        tx = self.rect.x + (self.rect.width - txt_surf.get_width()) // 2
        ty = self.rect.y + (self.rect.height - txt_surf.get_height()) // 2
        surface.blit(txt_surf, (tx, ty))

    def check_hover(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)

    def is_clicked(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)


def run_comparison_dashboard(
    model_path: str = "models/trained_q_table.pkl",
    window_width: int = 1040,
    window_height: int = 700,
    fps: int = 60,
    initial_scenario: str = "balanced",
) -> None:
    """
    Launches the live Pygame comparison dashboard with Start/Stop buttons.
    """
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Smart Traffic Light Controller — Fixed Timer vs RL Agent")
    clock = pygame.time.Clock()

    # Initialize fonts
    font_title = pygame.font.SysFont("Segoe UI", 17, bold=True)
    font_section = pygame.font.SysFont("Segoe UI", 13, bold=True)
    font_body = pygame.font.SysFont("Segoe UI", 11)
    font_stat = pygame.font.SysFont("Segoe UI", 15, bold=True)
    font_badge = pygame.font.SysFont("Segoe UI", 11, bold=True)
    font_btn = pygame.font.SysFont("Segoe UI", 13, bold=True)

    # Initialize agent and baseline controller
    agent = QLearningAgent()
    if not agent.load_model(model_path):
        print(f"⚠️ Model '{model_path}' not found or incompatible. Training model now...")
        agent, _ = train_agent(episodes=2500, model_save_path=model_path)

    fixed_controller = FixedTimerController(phase_duration=360)
    renderer = IntersectionRenderer(width=440, height=440)

    # Synchronized random seed
    current_seed = 42
    scenarios = ["balanced", "rush_hour", "asymmetric_ns", "asymmetric_ew", "light"]
    scenario_idx = scenarios.index(initial_scenario) if initial_scenario in scenarios else 0

    def init_environments(seed: int, scen: str):
        env_f = IntersectionEnv(max_steps=100000, seed=seed)
        env_f.set_traffic_scenario(scen)
        env_f.reset(seed=seed)

        env_r = IntersectionEnv(max_steps=100000, seed=seed)
        env_r.set_traffic_scenario(scen)
        env_r.reset(seed=seed)
        return env_f, env_r

    env_fixed, env_rl = init_environments(current_seed, scenarios[scenario_idx])

    # Simulation control states
    is_running = False  # Start in STOPPED state — user must click Start
    speed_multiplier = 1
    speed_options = [1, 2, 4, 8]
    speed_idx = 0
    message_banner = "Click ▶ START to begin the simulation"
    message_timer = 0

    # Create UI Buttons
    btn_y = 655
    btn_start = Button(30, btn_y, 100, 32, "▶ START", BTN_START_BG, BTN_START_HOVER)
    btn_stop = Button(140, btn_y, 100, 32, "■ STOP", BTN_STOP_BG, BTN_STOP_HOVER)
    btn_reset = Button(250, btn_y, 100, 32, "↺ RESET", BTN_RESET_BG, BTN_RESET_HOVER)

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()

        # Update button hover states
        btn_start.check_hover(mouse_pos)
        btn_stop.check_hover(mouse_pos)
        btn_reset.check_hover(mouse_pos)

        # 1. Handle Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_start.is_clicked(mouse_pos):
                    is_running = True
                    message_banner = "▶️ Simulation Running"
                    message_timer = 90

                elif btn_stop.is_clicked(mouse_pos):
                    is_running = False
                    message_banner = "⏹️ Simulation Stopped"
                    message_timer = 90

                elif btn_reset.is_clicked(mouse_pos):
                    current_seed += 100
                    env_fixed, env_rl = init_environments(current_seed, scenarios[scenario_idx])
                    is_running = False
                    message_banner = f"↺ Reset Simulation (Seed {current_seed}) — Click START"
                    message_timer = 150

            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    running = False

                elif event.key == pygame.K_SPACE:
                    is_running = not is_running
                    message_banner = "▶️ Simulation Running" if is_running else "⏹️ Simulation Stopped"
                    message_timer = 90

                elif event.key == pygame.K_UP:
                    speed_idx = min(len(speed_options) - 1, speed_idx + 1)
                    speed_multiplier = speed_options[speed_idx]
                    message_banner = f"⏩ Speed: {speed_multiplier}x"
                    message_timer = 90

                elif event.key == pygame.K_DOWN:
                    speed_idx = max(0, speed_idx - 1)
                    speed_multiplier = speed_options[speed_idx]
                    message_banner = f"⏩ Speed: {speed_multiplier}x"
                    message_timer = 90

                elif event.key == pygame.K_r:
                    current_seed += 100
                    env_fixed, env_rl = init_environments(current_seed, scenarios[scenario_idx])
                    is_running = False
                    message_banner = f"↺ Reset Simulation (Seed {current_seed}) — Click START"
                    message_timer = 120

                elif event.key == pygame.K_t:
                    scenario_idx = (scenario_idx + 1) % len(scenarios)
                    scen = scenarios[scenario_idx]
                    env_fixed.set_traffic_scenario(scen)
                    env_rl.set_traffic_scenario(scen)
                    message_banner = f"🚦 Traffic Scenario: {scen.replace('_', ' ').title()}"
                    message_timer = 120

                # --- Ambulance Triggers (1-4) ---
                elif event.key in [pygame.K_1, pygame.K_n]:
                    env_fixed.spawn_emergency_vehicle('N', VEHICLE_TYPE_AMBULANCE)
                    env_rl.spawn_emergency_vehicle('N', VEHICLE_TYPE_AMBULANCE)
                    message_banner = "🚑 Ambulance dispatched to NORTH lane!"
                    message_timer = 120

                elif event.key in [pygame.K_2, pygame.K_s]:
                    env_fixed.spawn_emergency_vehicle('S', VEHICLE_TYPE_AMBULANCE)
                    env_rl.spawn_emergency_vehicle('S', VEHICLE_TYPE_AMBULANCE)
                    message_banner = "🚑 Ambulance dispatched to SOUTH lane!"
                    message_timer = 120

                elif event.key in [pygame.K_3, pygame.K_e]:
                    env_fixed.spawn_emergency_vehicle('E', VEHICLE_TYPE_AMBULANCE)
                    env_rl.spawn_emergency_vehicle('E', VEHICLE_TYPE_AMBULANCE)
                    message_banner = "🚑 Ambulance dispatched to EAST lane!"
                    message_timer = 120

                elif event.key in [pygame.K_4, pygame.K_w]:
                    env_fixed.spawn_emergency_vehicle('W', VEHICLE_TYPE_AMBULANCE)
                    env_rl.spawn_emergency_vehicle('W', VEHICLE_TYPE_AMBULANCE)
                    message_banner = "🚑 Ambulance dispatched to WEST lane!"
                    message_timer = 120

                # --- Fire Truck Triggers (5-8) ---
                elif event.key == pygame.K_5:
                    env_fixed.spawn_emergency_vehicle('N', VEHICLE_TYPE_FIRETRUCK)
                    env_rl.spawn_emergency_vehicle('N', VEHICLE_TYPE_FIRETRUCK)
                    message_banner = "🚒 FIRE TRUCK (Tier 1) dispatched to NORTH lane!"
                    message_timer = 120

                elif event.key == pygame.K_6:
                    env_fixed.spawn_emergency_vehicle('S', VEHICLE_TYPE_FIRETRUCK)
                    env_rl.spawn_emergency_vehicle('S', VEHICLE_TYPE_FIRETRUCK)
                    message_banner = "🚒 FIRE TRUCK (Tier 1) dispatched to SOUTH lane!"
                    message_timer = 120

                elif event.key == pygame.K_7:
                    env_fixed.spawn_emergency_vehicle('E', VEHICLE_TYPE_FIRETRUCK)
                    env_rl.spawn_emergency_vehicle('E', VEHICLE_TYPE_FIRETRUCK)
                    message_banner = "🚒 FIRE TRUCK (Tier 1) dispatched to EAST lane!"
                    message_timer = 120

                elif event.key == pygame.K_8:
                    env_fixed.spawn_emergency_vehicle('W', VEHICLE_TYPE_FIRETRUCK)
                    env_rl.spawn_emergency_vehicle('W', VEHICLE_TYPE_FIRETRUCK)
                    message_banner = "🚒 FIRE TRUCK (Tier 1) dispatched to WEST lane!"
                    message_timer = 120

                # --- Simultaneous Conflicting Priority Test (F) ---
                elif event.key in [pygame.K_f, pygame.K_c]:
                    env_fixed.spawn_emergency_vehicle('N', VEHICLE_TYPE_FIRETRUCK)
                    env_fixed.spawn_emergency_vehicle('E', VEHICLE_TYPE_AMBULANCE)
                    env_rl.spawn_emergency_vehicle('N', VEHICLE_TYPE_FIRETRUCK)
                    env_rl.spawn_emergency_vehicle('E', VEHICLE_TYPE_AMBULANCE)
                    message_banner = "🔥 CONFLICT TEST: Fire Truck (North) vs Ambulance (East)!"
                    message_timer = 150

        # 2. Physics & Step Updates
        if is_running:
            for _ in range(speed_multiplier):
                act_f = fixed_controller.get_action(env_fixed)
                env_fixed.step(act_f)

                s_rl = env_rl.get_state()
                act_rl = agent.get_action(s_rl, env=env_rl, training=False)
                env_rl.step(act_rl)

        # 3. Rendering
        screen.fill(DARK_BG)

        # --- Top Header ---
        header_rect = pygame.Rect(0, 0, window_width, 48)
        pygame.draw.rect(screen, PANEL_BG, header_rect)
        pygame.draw.line(screen, PANEL_BORDER, (0, 48), (window_width, 48), 1)

        title_surf = font_title.render("Smart Traffic Light Controller — Multi-Tier Priority Preemption (Fixed Timer vs RL Agent)", True, TEXT_WHITE)
        screen.blit(title_surf, (20, 13))

        # Status indicator
        status_text = f"{'▶ RUNNING' if is_running else '⏹ STOPPED'} | {scenarios[scenario_idx].replace('_', ' ').title()} | Speed: {speed_multiplier}x"
        status_color = ACCENT_GREEN if is_running else ACCENT_RED
        mode_surf = font_badge.render(status_text, True, status_color)
        screen.blit(mode_surf, (window_width - mode_surf.get_width() - 20, 16))

        # --- Dual Viewports ---
        renderer.render(
            screen,
            env_fixed,
            offset_x=40,
            offset_y=56,
            title="BASELINE: Fixed-Timer Controller"
        )

        renderer.render(
            screen,
            env_rl,
            offset_x=560,
            offset_y=56,
            title="PROPOSED: Multi-Tier Q-Learning Agent"
        )

        # --- Bottom Metrics & Control Panel ---
        stat_y = 510
        panel_rect = pygame.Rect(20, stat_y, window_width - 40, 135)
        pygame.draw.rect(screen, PANEL_BG, panel_rect, border_radius=8)
        pygame.draw.rect(screen, PANEL_BORDER, panel_rect, width=1, border_radius=8)

        # Compute Metrics
        f_info = env_fixed._get_info()
        r_info = env_rl._get_info()

        f_wait = f_info['avg_wait_time']
        r_wait = r_info['avg_wait_time']
        wait_gain = ((f_wait - r_wait) / max(0.01, f_wait)) * 100 if f_wait > 0 else 0.0

        f_tp = f_info['throughput_per_min']
        r_tp = r_info['throughput_per_min']

        f_amb_w = f_info['avg_ambulance_wait_time']
        r_amb_w = r_info['avg_ambulance_wait_time']

        f_fire_w = f_info['avg_firetruck_wait_time']
        r_fire_w = r_info['avg_firetruck_wait_time']

        col_w = 235
        c1_x = 36
        c2_x = c1_x + col_w
        c3_x = c2_x + col_w
        c4_x = c3_x + col_w

        # Column 1: Average Wait Time & Throughput
        lbl1 = font_section.render("GENERAL METRICS", True, TEXT_MUTED)
        screen.blit(lbl1, (c1_x, stat_y + 10))
        f_w_txt = font_stat.render(f"Fixed: {f_wait:.1f}s | {f_tp:.1f} v/m", True, ACCENT_RED)
        r_w_txt = font_stat.render(f"RL:    {r_wait:.1f}s | {r_tp:.1f} v/m", True, ACCENT_GREEN)
        screen.blit(f_w_txt, (c1_x, stat_y + 32))
        screen.blit(r_w_txt, (c1_x, stat_y + 55))
        if wait_gain > 0:
            imp_badge = font_badge.render(f"↓ {wait_gain:.1f}% Wait Reduction", True, ACCENT_GREEN)
            screen.blit(imp_badge, (c1_x, stat_y + 80))

        # Column 2: Ambulance (Tier 2) Response
        lbl2 = font_section.render("🚑 AMBULANCE WAIT (Tier 2)", True, ACCENT_AMB)
        screen.blit(lbl2, (c2_x, stat_y + 10))
        f_amb_str = f"Fixed: {f_amb_w:.1f}s" if f_info['ambulance_cleared'] > 0 else "Fixed: —"
        r_amb_str = f"RL:    {r_amb_w:.1f}s" if r_info['ambulance_cleared'] > 0 else "RL:    —"
        f_amb_txt = font_stat.render(f_amb_str, True, ACCENT_RED if f_info['ambulance_cleared'] > 0 else TEXT_MUTED)
        r_amb_txt = font_stat.render(r_amb_str, True, ACCENT_GREEN if r_info['ambulance_cleared'] > 0 and r_amb_w < f_amb_w else ACCENT_AMB)
        screen.blit(f_amb_txt, (c2_x, stat_y + 32))
        screen.blit(r_amb_txt, (c2_x, stat_y + 55))
        amb_cnt_txt = font_body.render(f"Spawned: {r_info['ambulance_spawned']} | Cleared: {r_info['ambulance_cleared']}", True, TEXT_MUTED)
        screen.blit(amb_cnt_txt, (c2_x, stat_y + 80))

        # Column 3: Fire Truck (Tier 1) Response
        lbl3 = font_section.render("🚒 FIRE TRUCK WAIT (Tier 1)", True, ACCENT_FIRE)
        screen.blit(lbl3, (c3_x, stat_y + 10))
        f_fire_str = f"Fixed: {f_fire_w:.1f}s" if f_info['firetruck_cleared'] > 0 else "Fixed: —"
        r_fire_str = f"RL:    {r_fire_w:.1f}s" if r_info['firetruck_cleared'] > 0 else "RL:    —"
        f_fire_txt = font_stat.render(f_fire_str, True, ACCENT_RED if f_info['firetruck_cleared'] > 0 else TEXT_MUTED)
        r_fire_txt = font_stat.render(r_fire_str, True, ACCENT_GREEN if r_info['firetruck_cleared'] > 0 and r_fire_w < f_fire_w else ACCENT_FIRE)
        screen.blit(f_fire_txt, (c3_x, stat_y + 32))
        screen.blit(r_fire_txt, (c3_x, stat_y + 55))
        fire_cnt_txt = font_body.render(f"Spawned: {r_info['firetruck_spawned']} | Cleared: {r_info['firetruck_cleared']}", True, TEXT_MUTED)
        screen.blit(fire_cnt_txt, (c3_x, stat_y + 80))

        # Column 4: Controls & Conflict Key Guide
        lbl4 = font_section.render("KEYBOARD SHORTCUTS", True, ACCENT_ORANGE)
        screen.blit(lbl4, (c4_x, stat_y + 10))
        k1 = font_body.render("• [1, 2, 3, 4]: Ambulance (N/S/E/W)", True, TEXT_WHITE)
        k2 = font_body.render("• [5, 6, 7, 8]: Fire Truck (N/S/E/W)", True, TEXT_WHITE)
        k3 = font_body.render("• [F]: Conflict Test (Fire vs Amb)", True, ACCENT_FIRE)
        k4 = font_body.render("• [SPACE]: Start/Stop | [T]: Mode", True, TEXT_MUTED)
        screen.blit(k1, (c4_x, stat_y + 30))
        screen.blit(k2, (c4_x, stat_y + 48))
        screen.blit(k3, (c4_x, stat_y + 66))
        screen.blit(k4, (c4_x, stat_y + 84))

        # --- Bottom Button Bar ---
        btn_bar_rect = pygame.Rect(0, 648, window_width, 52)
        pygame.draw.rect(screen, PANEL_BG, btn_bar_rect)
        pygame.draw.line(screen, PANEL_BORDER, (0, 648), (window_width, 648), 1)

        btn_start.draw(screen, font_btn)
        btn_stop.draw(screen, font_btn)
        btn_reset.draw(screen, font_btn)

        # Bottom Notification Banner (right of buttons)
        if message_timer > 0:
            message_timer -= 1
        banner_color = ACCENT_ORANGE if ("Ambulance" in message_banner or "FIRE" in message_banner or "CONFLICT" in message_banner) else TEXT_MUTED
        if "Running" in message_banner:
            banner_color = ACCENT_GREEN
        elif "Stopped" in message_banner or "Reset" in message_banner:
            banner_color = ACCENT_RED
        banner_surf = font_body.render(message_banner, True, banner_color)
        screen.blit(banner_surf, (370, btn_y + 8))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    run_comparison_dashboard()
