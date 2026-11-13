import time
import config
from typing import Tuple

from goggle_state import WearState

try:
    import board
    import neopixel_spi
except ImportError:
    board = None
    neopixel_spi = None


class LEDDisplay:
    """
    Simple RGB LED controller for a 40-pixel addressable LED strip.

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

        if board is None or neopixel_spi is None:
            raise ImportError(
                "CircuitPython NeoPixel SPI dependencies are not installed. "
                "Install adafruit-circuitpython-neopixel-spi and Adafruit Blinka, "
                "or enable LED_SIMULATE."
            )

        self.strip = neopixel_spi.NeoPixel_SPI(
            board.SPI(),
            config.LED_PIXEL_COUNT,
            auto_write=False,
            brightness=config.LED_BRIGHTNESS / 255,
            pixel_order=config.LED_PIXEL_ORDER,
        )
        self.clear()

    def set_pixel(self, index: int, rgb: Tuple[int, int, int]):
        if not 0 <= index < config.LED_PIXEL_COUNT:
            return

        if config.LED_SIMULATE:
            return

        self.strip[index] = rgb
        self.strip.show()

    def fill(self, rgb: Tuple[int, int, int]):
        if config.LED_SIMULATE:
            print(f"[LED] fill -> {rgb}")
            return

        self.strip.fill(rgb)
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
        if config.LED_SIMULATE:
            self.clear()
            return

        if self.strip is not None:
            self.strip.deinit()
            self.strip = None
        self.current_state = WearState.UNKNOWN