"""Sound module modified to disable all audio output by default.

Reason: Disable audio during machine learning runs to avoid interference.
This module preserves the original API but makes `play_sfx` and
`play_music` no-ops and avoids importing or initializing the mixer.
"""

class Sound:
    def __init__(self, enabled: bool = False):
        """Create a Sound object.

        The `enabled` argument is accepted for API compatibility but audio is
        disabled unconditionally to prevent any sound output during ML runs.
        """
        # Force-disable audio regardless of the caller's request
        self.enabled = False

        # Provide attributes expected by the codebase so callers won't fail
        self.music_channel = DummyChannel()
        self.sfx_channel = DummyChannel()
        self.allowSFX = False

        # Sound asset placeholders (kept for compatibility)
        self.soundtrack = None
        self.coin = None
        self.bump = None
        self.stomp = None
        self.jump = None
        self.death = None
        self.kick = None
        self.brick_bump = None
        self.powerup = None
        self.powerup_appear = None
        self.pipe = None

    def play_sfx(self, sfx, *args, **kwargs):
        """No-op for playing sound effects."""
        return None

    def play_music(self, music, *args, **kwargs):
        """No-op for playing music."""
        return None


class DummyChannel:
    """A dummy audio channel that implements the parts of the pygame
    channel API used by the project but does nothing."""

    def play(self, sound, loops=0):
        return None

    def stop(self):
        return None

    def set_volume(self, volume):
        return None

    def get_busy(self):
        return False
