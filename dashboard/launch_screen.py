"""
Animated Pygame Launch Screen for the Smart Traffic Light RL Project.
Shows project branding, team info, scenario selector, and animated traffic light.
"""
import sys
import os
import math
import pygame

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DARK_BG      = (18,  22,  28)
PANEL_BG     = (26,  31,  38)
PANEL_BORDER = (48,  54,  61)
TEXT_WHITE   = (240, 246, 252)
TEXT_MUTED   = (139, 148, 158)
ACCENT_BLUE  = (88,  166, 255)
ACCENT_GREEN = (63,  185, 80)
ACCENT_ORANGE= (210, 153, 34)
ACCENT_RED   = (248, 81,  73)
ACCENT_FIRE  = (255, 75,  75)
TL_RED_ON    = (220, 50,  50)
TL_YELLOW_ON = (220, 200, 50)
TL_GREEN_ON  = (50,  200, 80)
TL_OFF       = (30,  35,  42)

SCENARIOS = [
    ("balanced",      "Balanced Traffic",        "Equal flow on all 4 lanes"),
    ("rush_hour",     "Rush Hour",               "High density - peak congestion"),
    ("asymmetric_ns", "Asymmetric (NS Heavy)",   "North-South lanes overloaded"),
    ("asymmetric_ew", "Asymmetric (EW Heavy)",   "East-West lanes overloaded"),
    ("light",         "Light Traffic",           "Low density - off-peak hours"),
]


def _draw_traffic_light(surface, cx, cy, phase_frame, radius=14, pole_h=60):
    pygame.draw.rect(surface, (60, 65, 72), (cx - 4, cy - pole_h, 8, pole_h + radius * 3 + 16))
    housing_rect = pygame.Rect(cx - radius - 6, cy - pole_h, (radius + 6) * 2, radius * 3 * 2 + 20)
    pygame.draw.rect(surface, (38, 43, 50), housing_rect, border_radius=8)
    pygame.draw.rect(surface, (55, 60, 68), housing_rect, width=2, border_radius=8)
    phase_frame = phase_frame % 180
    red_on    = phase_frame < 90
    yellow_on = 90 <= phase_frame < 120
    green_on  = phase_frame >= 120
    bulb_y_start = cy - pole_h + 10
    for i, (flag, col_on) in enumerate([(red_on, TL_RED_ON), (yellow_on, TL_YELLOW_ON), (green_on, TL_GREEN_ON)]):
        by = bulb_y_start + i * (radius * 2 + 4)
        color = col_on if flag else TL_OFF
        pygame.draw.circle(surface, color, (cx, by + radius), radius)
        if flag:
            pygame.draw.circle(surface, color, (cx, by + radius), radius + 4, 2)


def run_launch_screen(window_width: int = 1040, window_height: int = 700):
    """Display animated launch screen. Returns (scenario_key, seed) or None."""
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Smart Traffic Light Controller - RL Project")
    clock = pygame.time.Clock()

    font_h1    = pygame.font.SysFont("Segoe UI", 28, bold=True)
    font_h2    = pygame.font.SysFont("Segoe UI", 17, bold=True)
    font_body  = pygame.font.SysFont("Segoe UI", 14)
    font_small = pygame.font.SysFont("Segoe UI", 12)
    font_badge = pygame.font.SysFont("Segoe UI", 11, bold=True)
    font_scen  = pygame.font.SysFont("Segoe UI", 15, bold=True)
    font_desc  = pygame.font.SysFont("Segoe UI", 12)

    selected_idx = 0
    frame        = 0
    tl_frame     = 0

    running = True
    while running:
        clock.tick(60)
        frame    += 1
        tl_frame += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    return None
                elif event.key in (pygame.K_UP,):
                    selected_idx = (selected_idx - 1) % len(SCENARIOS)
                elif event.key in (pygame.K_DOWN,):
                    selected_idx = (selected_idx + 1) % len(SCENARIOS)
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                    pygame.quit()
                    return SCENARIOS[selected_idx][0], 42

        screen.fill(DARK_BG)
        for gx in range(0, window_width, 60):
            pygame.draw.line(screen, (24, 28, 35), (gx, 0), (gx, window_height))
        for gy in range(0, window_height, 60):
            pygame.draw.line(screen, (24, 28, 35), (0, gy), (window_width, gy))

        _draw_traffic_light(screen, 80,  200, tl_frame,       radius=12, pole_h=50)
        _draw_traffic_light(screen, 960, 200, tl_frame + 60,  radius=12, pole_h=50)
        _draw_traffic_light(screen, 80,  480, tl_frame + 120, radius=12, pole_h=50)
        _draw_traffic_light(screen, 960, 480, tl_frame + 90,  radius=12, pole_h=50)

        pygame.draw.rect(screen, PANEL_BG, (0, 0, window_width, 56))
        pygame.draw.line(screen, ACCENT_BLUE, (0, 56), (window_width, 56), 2)
        dept = font_small.render(
            "Reinforcement Learning Research Project  |  Tabular Q-Learning  |  Multi-Tier Priority",
            True, TEXT_MUTED
        )
        screen.blit(dept, ((window_width - dept.get_width()) // 2, 20))

        title1 = font_h1.render("Smart Traffic Light Controller", True, TEXT_WHITE)
        title2 = font_h2.render("Using Reinforcement Learning (Q-Learning)", True, ACCENT_BLUE)
        screen.blit(title1, ((window_width - title1.get_width()) // 2, 80))
        screen.blit(title2, ((window_width - title2.get_width()) // 2, 116))

        pygame.draw.line(screen, PANEL_BORDER, ((window_width-600)//2, 152), ((window_width+600)//2, 152), 1)

        card_w, card_h = 720, 80
        card_x = (window_width - card_w) // 2
        card_rect = pygame.Rect(card_x, 164, card_w, card_h)
        pygame.draw.rect(screen, PANEL_BG, card_rect, border_radius=10)
        pygame.draw.rect(screen, PANEL_BORDER, card_rect, width=1, border_radius=10)
        desc_lbl = font_badge.render("ABOUT", True, ACCENT_ORANGE)
        screen.blit(desc_lbl, (card_x + 20, 174))
        desc1 = font_body.render("Adaptive traffic signal control using Tabular Q-Learning with multi-tier", True, TEXT_WHITE)
        desc2 = font_body.render("emergency vehicle priority (Fire Trucks + Ambulances). Calibrated with Kaggle data.", True, TEXT_MUTED)
        screen.blit(desc1, (card_x + 20, 196))
        screen.blit(desc2, (card_x + 20, 216))

        badges = [
            ("Q-Learning Agent",      ACCENT_BLUE),
            ("Fixed-Timer Baseline",  ACCENT_RED),
            ("Multi-Tier Priority",   ACCENT_FIRE),
            ("Real-time Simulation",  ACCENT_GREEN),
            ("Kaggle Dataset",        ACCENT_ORANGE),
        ]
        bx = (window_width - (len(badges) * 148 - 8)) // 2
        for label, color in badges:
            brect = pygame.Rect(bx, 288, 140, 28)
            pygame.draw.rect(screen, PANEL_BG, brect, border_radius=5)
            pygame.draw.rect(screen, color, brect, width=1, border_radius=5)
            bsurf = font_badge.render(label, True, color)
            screen.blit(bsurf, (bx + (140 - bsurf.get_width()) // 2, 296))
            bx += 148

        sel_lbl = font_badge.render("SELECT TRAFFIC SCENARIO  (UP/DOWN arrows, then ENTER to launch)", True, TEXT_MUTED)
        screen.blit(sel_lbl, ((window_width - sel_lbl.get_width()) // 2, 336))

        box_w, box_h = 620, 46
        box_x = (window_width - box_w) // 2
        for i, (key, name, desc) in enumerate(SCENARIOS):
            by = 360 + i * (box_h + 6)
            is_sel = (i == selected_idx)
            bg_col = (36, 58, 80) if is_sel else PANEL_BG
            bd_col = ACCENT_BLUE if is_sel else PANEL_BORDER
            box_rect = pygame.Rect(box_x, by, box_w, box_h)
            pygame.draw.rect(screen, bg_col, box_rect, border_radius=7)
            pygame.draw.rect(screen, bd_col, box_rect, width=2 if is_sel else 1, border_radius=7)
            if is_sel:
                arrow = font_scen.render(">", True, ACCENT_BLUE)
                screen.blit(arrow, (box_x + 12, by + (box_h - arrow.get_height()) // 2))
            screen.blit(font_scen.render(name, True, TEXT_WHITE if is_sel else TEXT_MUTED), (box_x + 36, by + 6))
            screen.blit(font_desc.render(desc, True, ACCENT_BLUE if is_sel else (80, 88, 96)), (box_x + 36, by + 26))

        alpha  = int(160 + 95 * math.sin(frame * 0.07))
        g_col  = min(185, int(80 * alpha / 255))
        prompt = font_h2.render("Press  ENTER  to Launch Simulation", True, (0, g_col, 0))
        prompt = font_h2.render("Press  ENTER  to Launch Simulation", True, ACCENT_GREEN)
        screen.blit(prompt, ((window_width - prompt.get_width()) // 2, 644))

        pygame.display.flip()

    pygame.quit()
    return None
