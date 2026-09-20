"""
Pygame-based graphical renderer for the 4-way intersection.
Renders asphalt roads, crosswalks, vehicles, traffic lights, and live metrics.
"""
import pygame
import math
from typing import Tuple
from environment.intersection_env import (
    IntersectionEnv,
    PHASE_NS_GREEN,
    PHASE_EW_GREEN,
    PHASE_NS_YELLOW,
    PHASE_EW_YELLOW,
)
from environment.vehicle import Vehicle


# Theme Colors
COLOR_BG = (28, 30, 38)
COLOR_GRASS = (40, 48, 56)
COLOR_ROAD = (48, 54, 61)
COLOR_MARKING = (240, 240, 240)
COLOR_YELLOW_LINE = (245, 183, 33)
COLOR_STOP_LINE = (255, 255, 255)
COLOR_CROSSWALK = (200, 205, 215)
COLOR_BOX = (22, 27, 34)

# Traffic Light Colors
COLOR_RED_ON = (255, 75, 75)
COLOR_RED_OFF = (80, 20, 20)
COLOR_YELLOW_ON = (255, 205, 50)
COLOR_YELLOW_OFF = (80, 65, 15)
COLOR_GREEN_ON = (46, 213, 115)
COLOR_GREEN_OFF = (15, 65, 35)

# Text & Accent Colors
COLOR_TEXT = (240, 246, 252)
COLOR_TEXT_DIM = (139, 148, 158)
COLOR_ACCENT_BLUE = (88, 166, 255)
COLOR_ACCENT_GREEN = (63, 185, 80)
COLOR_ACCENT_ORANGE = (210, 153, 34)
COLOR_ACCENT_RED = (248, 81, 73)


class IntersectionRenderer:
    """
    Renders an IntersectionEnv instance to a Pygame Surface.
    """

    def __init__(self, width: int = 420, height: int = 420):
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2

        self.road_width = 84
        self.lane_width = self.road_width // 2

        # Initialize fonts if pygame is initialized
        self.font_small = None
        self.font_mid = None
        self.font_bold = None
        self._init_fonts()

    def _init_fonts(self) -> None:
        if pygame.font.get_init():
            self.font_small = pygame.font.SysFont("Segoe UI", 12)
            self.font_mid = pygame.font.SysFont("Segoe UI", 14, bold=True)
            self.font_bold = pygame.font.SysFont("Segoe UI", 18, bold=True)

    def render(self, surface: pygame.Surface, env: IntersectionEnv, offset_x: int = 0, offset_y: int = 0, title: str = "") -> None:
        """
        Renders the intersection environment onto the specified surface.
        """
        if self.font_small is None:
            self._init_fonts()

        # 1. Background / Grass
        rect = pygame.Rect(offset_x, offset_y, self.width, self.height)
        pygame.draw.rect(surface, COLOR_GRASS, rect)

        # 2. Roads
        # North-South Road
        ns_road = pygame.Rect(
            offset_x + self.center_x - self.road_width // 2,
            offset_y,
            self.road_width,
            self.height
        )
        pygame.draw.rect(surface, COLOR_ROAD, ns_road)

        # East-West Road
        ew_road = pygame.Rect(
            offset_x,
            offset_y + self.center_y - self.road_width // 2,
            self.width,
            self.road_width
        )
        pygame.draw.rect(surface, COLOR_ROAD, ew_road)

        # 3. Road Markings & Zebra Crossings
        self._draw_road_markings(surface, offset_x, offset_y)

        # 4. Traffic Lights
        self._draw_traffic_lights(surface, env, offset_x, offset_y)

        # 5. Vehicles
        self._draw_vehicles(surface, env, offset_x, offset_y)

        # 6. Lane Queue Badges
        self._draw_queue_badges(surface, env, offset_x, offset_y)

        # 7. Title Header
        if title and self.font_bold:
            header_surf = self.font_bold.render(title, True, COLOR_TEXT)
            surface.blit(header_surf, (offset_x + 15, offset_y + 12))

    def _draw_road_markings(self, surface: pygame.Surface, ox: int, oy: int) -> None:
        cx, cy = ox + self.center_x, oy + self.center_y
        hw = self.road_width // 2

        # Center yellow dividing lines (dashed)
        dash_len = 10
        gap_len = 8

        # North road center line
        y = oy
        while y < cy - hw - 20:
            pygame.draw.line(surface, COLOR_YELLOW_LINE, (cx, y), (cx, min(y + dash_len, cy - hw - 20)), 2)
            y += dash_len + gap_len

        # South road center line
        y = cy + hw + 20
        while y < oy + self.height:
            pygame.draw.line(surface, COLOR_YELLOW_LINE, (cx, y), (cx, min(y + dash_len, oy + self.height)), 2)
            y += dash_len + gap_len

        # West road center line
        x = ox
        while x < cx - hw - 20:
            pygame.draw.line(surface, COLOR_YELLOW_LINE, (x, cy), (min(x + dash_len, cx - hw - 20), cy), 2)
            x += dash_len + gap_len

        # East road center line
        x = cx + hw + 20
        while x < ox + self.width:
            pygame.draw.line(surface, COLOR_YELLOW_LINE, (x, cy), (min(x + dash_len, ox + self.width), cy), 2)
            x += dash_len + gap_len

        # Stop lines (Solid White)
        stop_offset = hw + 8
        # North incoming (left lane of NS road) stop line
        pygame.draw.line(surface, COLOR_STOP_LINE, (cx - hw, cy - stop_offset), (cx, cy - stop_offset), 4)
        # South incoming (right lane of NS road) stop line
        pygame.draw.line(surface, COLOR_STOP_LINE, (cx, cy + stop_offset), (cx + hw, cy + stop_offset), 4)
        # West incoming (top lane of EW road) stop line
        pygame.draw.line(surface, COLOR_STOP_LINE, (cx - stop_offset, cy), (cx - stop_offset, cy + hw), 4)
        # East incoming (bottom lane of EW road) stop line
        pygame.draw.line(surface, COLOR_STOP_LINE, (cx + stop_offset, cy - hw), (cx + stop_offset, cy), 4)

    def _draw_traffic_lights(self, surface: pygame.Surface, env: IntersectionEnv, ox: int, oy: int) -> None:
        cx, cy = ox + self.center_x, oy + self.center_y
        hw = self.road_width // 2

        # Determine light states
        sig = env.signal_state
        # We need to import PHASE_ALL_RED or use number 4
        # Since I'll import it or check if it's 4
        
        ns_red = (sig in [PHASE_EW_GREEN, PHASE_EW_YELLOW, 4])
        ns_yellow = (sig == PHASE_NS_YELLOW)
        ns_green = (sig == PHASE_NS_GREEN)

        ew_red = (sig in [PHASE_NS_GREEN, PHASE_NS_YELLOW, 4])
        ew_yellow = (sig == PHASE_EW_YELLOW)
        ew_green = (sig == PHASE_EW_GREEN)

        # Draw 4 Signal Housings
        # North signal (controls vehicles heading South)
        self._draw_signal_box(surface, cx - hw - 22, cy - hw - 36, ns_red, ns_yellow, ns_green)
        # South signal (controls vehicles heading North)
        self._draw_signal_box(surface, cx + hw + 6, cy + hw + 4, ns_red, ns_yellow, ns_green)
        # West signal (controls vehicles heading East)
        self._draw_signal_box(surface, cx - hw - 36, cy + hw + 6, ew_red, ew_yellow, ew_green, horizontal=True)
        # East signal (controls vehicles heading West)
        self._draw_signal_box(surface, cx + hw + 6, cy - hw - 22, ew_red, ew_yellow, ew_green, horizontal=True)

    def _draw_signal_box(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        is_red: bool,
        is_yellow: bool,
        is_green: bool,
        horizontal: bool = False
    ) -> None:
        radius = 4
        if not horizontal:
            box = pygame.Rect(x, y, 16, 32)
            pygame.draw.rect(surface, COLOR_BOX, box, border_radius=4)
            pygame.draw.rect(surface, (100, 110, 120), box, width=1, border_radius=4)

            # Red
            c_r = COLOR_RED_ON if is_red else COLOR_RED_OFF
            pygame.draw.circle(surface, c_r, (x + 8, y + 6), radius)
            # Yellow
            c_y = COLOR_YELLOW_ON if is_yellow else COLOR_YELLOW_OFF
            pygame.draw.circle(surface, c_y, (x + 8, y + 16), radius)
            # Green
            c_g = COLOR_GREEN_ON if is_green else COLOR_GREEN_OFF
            pygame.draw.circle(surface, c_g, (x + 8, y + 26), radius)
        else:
            box = pygame.Rect(x, y, 32, 16)
            pygame.draw.rect(surface, COLOR_BOX, box, border_radius=4)
            pygame.draw.rect(surface, (100, 110, 120), box, width=1, border_radius=4)

            # Red
            c_r = COLOR_RED_ON if is_red else COLOR_RED_OFF
            pygame.draw.circle(surface, c_r, (x + 6, y + 8), radius)
            # Yellow
            c_y = COLOR_YELLOW_ON if is_yellow else COLOR_YELLOW_OFF
            pygame.draw.circle(surface, c_y, (x + 16, y + 8), radius)
            # Green
            c_g = COLOR_GREEN_ON if is_green else COLOR_GREEN_OFF
            pygame.draw.circle(surface, c_g, (x + 26, y + 8), radius)

    def _get_vehicle_coords(self, veh: Vehicle, ox: int, oy: int) -> Tuple[float, float, float, float]:
        """Maps 1D lane position to 2D (x, y, width, height) screen coordinates."""
        cx, cy = ox + self.center_x, oy + self.center_y
        lane_w = self.lane_width

        # North incoming: starts at top (y=0), moves down towards +y
        if veh.lane == 'N':
            x = cx - lane_w + (lane_w - veh.width) / 2
            y = oy + veh.position
            return x, y, veh.width, veh.length

        # South incoming: starts at bottom (y=height), moves up towards -y
        elif veh.lane == 'S':
            x = cx + (lane_w - veh.width) / 2
            y = oy + self.height - veh.position - veh.length
            return x, y, veh.width, veh.length

        # West incoming: starts at left (x=0), moves right towards +x
        elif veh.lane == 'W':
            x = ox + veh.position
            y = cy + (lane_w - veh.width) / 2
            return x, y, veh.length, veh.width

        # East incoming: starts at right (x=width), moves left towards -x
        elif veh.lane == 'E':
            x = ox + self.width - veh.position - veh.length
            y = cy - lane_w + (lane_w - veh.width) / 2
            return x, y, veh.length, veh.width

        return 0, 0, 0, 0

    def _draw_vehicles(self, surface: pygame.Surface, env: IntersectionEnv, ox: int, oy: int) -> None:
        step = env.current_step

        for veh_list in env.lanes.values():
            for veh in veh_list:
                x, y, w, h = self._get_vehicle_coords(veh, ox, oy)
                veh_rect = pygame.Rect(int(x), int(y), int(w), int(h))

                if veh.vehicle_type == 2:  # VEHICLE_TYPE_FIRETRUCK
                    # 🚒 Fire Truck Rendering
                    pygame.draw.rect(surface, (210, 20, 20), veh_rect, border_radius=4)
                    pygame.draw.rect(surface, (140, 10, 10), veh_rect, width=2, border_radius=4)

                    # Metallic Roof Ladder & Equipment Panel
                    is_vertical = veh.lane in ['N', 'S']
                    if is_vertical:
                        ladder_rect = pygame.Rect(int(x + 3), int(y + h * 0.25), int(w - 6), int(h * 0.50))
                        pygame.draw.rect(surface, (190, 195, 205), ladder_rect, border_radius=2)
                        # Ladder rungs
                        for rung_y in range(int(y + h * 0.30), int(y + h * 0.70), 4):
                            pygame.draw.line(surface, (100, 105, 115), (int(x + 4), rung_y), (int(x + w - 5), rung_y), 1)
                    else:
                        ladder_rect = pygame.Rect(int(x + w * 0.25), int(y + 3), int(w * 0.50), int(h - 6))
                        pygame.draw.rect(surface, (190, 195, 205), ladder_rect, border_radius=2)
                        # Ladder rungs
                        for rung_x in range(int(x + w * 0.30), int(x + w * 0.70), 4):
                            pygame.draw.line(surface, (100, 105, 115), (rung_x, int(y + 4)), (rung_x, int(y + h - 5)), 1)

                    # Front Windshield
                    if is_vertical:
                        front_y = int(y + h - 7) if veh.lane == 'N' else int(y + 2)
                        windshield = pygame.Rect(int(x + 2), front_y, int(w - 4), 5)
                    else:
                        front_x = int(x + w - 7) if veh.lane == 'W' else int(x + 2)
                        windshield = pygame.Rect(front_x, int(y + 2), 5, int(h - 4))
                    pygame.draw.rect(surface, (35, 45, 55), windshield, border_radius=2)

                    # Animated Dual Emergency Strobes (Alternating Crimson & Amber)
                    is_pulse = (step // 5) % 2 == 0
                    strobe_1 = (255, 30, 30) if is_pulse else (255, 200, 0)
                    strobe_2 = (255, 200, 0) if is_pulse else (255, 30, 30)

                    if is_vertical:
                        sy = int(y + h - 3) if veh.lane == 'N' else int(y + 3)
                        pygame.draw.circle(surface, strobe_1, (int(x + 3), sy), 3)
                        pygame.draw.circle(surface, strobe_2, (int(x + w - 4), sy), 3)
                    else:
                        sx = int(x + w - 3) if veh.lane == 'W' else int(x + 3)
                        pygame.draw.circle(surface, strobe_1, (sx, int(y + 3)), 3)
                        pygame.draw.circle(surface, strobe_2, (sx, int(y + h - 4)), 3)

                elif veh.vehicle_type == 1:  # VEHICLE_TYPE_AMBULANCE
                    # 🚑 Ambulance Rendering
                    pygame.draw.rect(surface, (255, 255, 255), veh_rect, border_radius=4)
                    pygame.draw.rect(surface, (200, 30, 30), veh_rect, width=2, border_radius=4)

                    # Red Cross in center
                    cross_cx = int(x + w / 2)
                    cross_cy = int(y + h / 2)
                    pygame.draw.line(surface, (220, 20, 20), (cross_cx - 4, cross_cy), (cross_cx + 4, cross_cy), 2)
                    pygame.draw.line(surface, (220, 20, 20), (cross_cx, cross_cy - 4), (cross_cx, cross_cy + 4), 2)

                    # Animated Flashing Siren (Red & Blue)
                    is_pulse = (step // 6) % 2 == 0
                    siren_color_1 = (255, 50, 50) if is_pulse else (30, 144, 255)
                    siren_color_2 = (30, 144, 255) if is_pulse else (255, 50, 50)

                    if veh.lane in ['N', 'S']:
                        pygame.draw.circle(surface, siren_color_1, (cross_cx - 3, int(y + 4)), 3)
                        pygame.draw.circle(surface, siren_color_2, (cross_cx + 3, int(y + 4)), 3)
                    else:
                        pygame.draw.circle(surface, siren_color_1, (int(x + 4), cross_cy - 3), 3)
                        pygame.draw.circle(surface, siren_color_2, (int(x + 4), cross_cy + 3), 3)

                else:
                    # Standard sleek civilian vehicle
                    pygame.draw.rect(surface, veh.color, veh_rect, border_radius=3)
                    # Windshield tint
                    tint_rect = pygame.Rect(int(x + w * 0.15), int(y + h * 0.2), max(1, int(w * 0.7)), max(1, int(h * 0.6)))
                    pygame.draw.rect(surface, (30, 35, 42), tint_rect, border_radius=2)

                    # Brake lights indicator
                    if veh.is_braking:
                        if veh.lane == 'N':
                            pygame.draw.circle(surface, (255, 30, 30), (int(x + 2), int(y + 2)), 2)
                            pygame.draw.circle(surface, (255, 30, 30), (int(x + w - 3), int(y + 2)), 2)
                        elif veh.lane == 'S':
                            pygame.draw.circle(surface, (255, 30, 30), (int(x + 2), int(y + h - 3)), 2)
                            pygame.draw.circle(surface, (255, 30, 30), (int(x + w - 3), int(y + h - 3)), 2)
                        elif veh.lane == 'W':
                            pygame.draw.circle(surface, (255, 30, 30), (int(x + 2), int(y + 2)), 2)
                            pygame.draw.circle(surface, (255, 30, 30), (int(x + 2), int(y + h - 3)), 2)
                        elif veh.lane == 'E':
                            pygame.draw.circle(surface, (255, 30, 30), (int(x + w - 3), int(y + 2)), 2)
                            pygame.draw.circle(surface, (255, 30, 30), (int(x + w - 3), int(y + h - 3)), 2)

    def _draw_queue_badges(self, surface: pygame.Surface, env: IntersectionEnv, ox: int, oy: int) -> None:
        """Renders small badge tags showing vehicle queue counts per direction."""
        if not self.font_small:
            return

        cx, cy = ox + self.center_x, oy + self.center_y
        counts = env.get_queue_counts()

        badges = [
            ('N', cx - 72, cy - self.road_width - 18),
            ('S', cx + 24, cy + self.road_width + 6),
            ('W', cx - self.road_width - 56, cy + 24),
            ('E', cx + self.road_width + 12, cy - 38),
        ]

        for lane, bx, by in badges:
            c = counts[lane]
            has_fire = any(v.vehicle_type == 2 and v.is_in_queue(env.STOP_LINE_POS) for v in env.lanes[lane])
            has_amb = any(v.vehicle_type == 1 and v.is_in_queue(env.STOP_LINE_POS) for v in env.lanes[lane])

            if has_fire:
                text = f"{lane}: {c} [FIRE!]"
                badge_color = (255, 60, 60)
            elif has_amb:
                text = f"{lane}: {c} [AMB]"
                badge_color = (60, 180, 255)
            else:
                text = f"{lane}: {c}"
                badge_color = COLOR_ACCENT_GREEN if c == 0 else (COLOR_ACCENT_ORANGE if c <= 2 else COLOR_ACCENT_RED)

            txt_surf = self.font_small.render(text, True, (255, 255, 255))
            pad = 3
            bg_rect = pygame.Rect(bx - pad, by - pad, txt_surf.get_width() + pad * 2, txt_surf.get_height() + pad * 2)
            pygame.draw.rect(surface, (25, 30, 36), bg_rect, border_radius=4)
            pygame.draw.rect(surface, badge_color, bg_rect, width=1, border_radius=4)
            surface.blit(txt_surf, (bx, by))
