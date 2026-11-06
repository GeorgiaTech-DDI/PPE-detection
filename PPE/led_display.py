import time
import config
from typing import Tuple

from goggle_state import WearState

try:
    from rpi_ws281x import PixelStrip, Color
except ImportError:
    PixelStrip = None
    Color = None


class LEDDisplay:
    """
    Simple RGB LED controller for an addressable 8x32 display or strip.

    Designed to be easy to expand:
    - set solid status colors
    - draw individual pixels
    - add animations later
    - add text/icons later
    """

    DEFAULT_COLORS = {
        WearState.UNKNOWN: (0, 0, 0),
        WearState.WEARING: (0, 255, 0),       # green
        WearState.NOT_WEARING: (255, 0, 0),     # red
    }

    def __init__(self):
        self.strip = None
        self.current_state = WearState.UNKNOWN

    def begin(self):
        if config.LED_SIMULATE:
            print("[LED] Simulation mode enabled")
            return

        if PixelStrip is None:
            raise ImportError(
                "rpi_ws281x is not installed. Install it or enable simulate=True."
            )

        self.strip = PixelStrip(
            config.LED_WIDTH * config.LED_HEIGHT,
            config.LED_PIN,
            config.LED_FREQUENCY_HZ,
            config.LED_DMA,
            config.LED_INVERT,
            config.LED_BRIGHTNESS,
            config.LED_CHANNEL,
        )
        self.strip.begin()
        self.clear()

    def _color(self, rgb: Tuple[int, int, int]):
        r, g, b = rgb
        if Color is None:
            return (r, g, b)
        # Most WS2812 setups on Pi use GRB order in practice.
        if config.LED_PIXEL_ORDER.upper() == "GRB":
            return Color(g, r, b)
        return Color(r, g, b)

    def xy_to_index(self, x: int, y: int) -> int:
        """
        Serpentine mapping for 8x32 matrix.
        Change this if your panel wiring is different.
        """
        if y % 2 == 0:
            return y * config.LED_WIDTH + x
        return y * config.LED_WIDTH + (config.LED_WIDTH - 1 - x)

    def set_pixel(self, x: int, y: int, rgb: Tuple[int, int, int]):
        if not (0 <= x < config.LED_WIDTH and 0 <= y < config.LED_HEIGHT):
            return

        idx = self.xy_to_index(x, y)
        if config.LED_SIMULATE:
            return

        self.strip.setPixelColor(idx, self._color(rgb))

    def fill(self, rgb: Tuple[int, int, int]):
        if config.LED_SIMULATE:
            print(f"[LED] fill -> {rgb}")
            return

        color = self._color(rgb)
        for i in range(config.LED_WIDTH * config.LED_HEIGHT):
            self.strip.setPixelColor(i, color)
        self.strip.show()

    def clear(self):
        self.fill((0, 0, 0))
        self.current_state = WearState.UNKNOWN

    def set_state(self, state: WearState):
        """
        Set a named state color.
        Avoids re-sending the same color repeatedly.
        """
        if state == self.current_state:
            return

        if state not in self.DEFAULT_COLORS:
            raise ValueError(f"Unknown LED state: {state}")

        self.fill(self.DEFAULT_COLORS[state])
        self.current_state = state

    def flash(self, rgb: Tuple[int, int, int], times: int = 3, delay: float = 0.2):
        for _ in range(times):
            self.fill(rgb)
            time.sleep(delay)
            self.clear()
            time.sleep(delay)

    def test_cycle(self):
        self.fill((255, 0, 0))
        time.sleep(0.5)
        self.fill((0, 255, 0))
        time.sleep(0.5)
        self.fill((0, 0, 255))
        time.sleep(0.5)
        self.clear()

    def cleanup(self):
        self.clear()