from pygame import mixer


class Sound:
    def __init__(self, enabled=True):
        """
        Sound クラス
        
        Parameters:
        -----------
        enabled : bool
            True: 通常モード（音声出力）
            False: headless モード（音声なし）
        """
        self.enabled = enabled
        
        if enabled:
            self.music_channel = mixer.Channel(0)
            self.music_channel.set_volume(0.2)
            self.sfx_channel = mixer.Channel(1)
            self.sfx_channel.set_volume(0.2)

            self.allowSFX = True

            self.soundtrack = mixer.Sound("./sfx/main_theme.ogg")
            self.coin = mixer.Sound("./sfx/coin.ogg")
            self.bump = mixer.Sound("./sfx/bump.ogg")
            self.stomp = mixer.Sound("./sfx/stomp.ogg")
            self.jump = mixer.Sound("./sfx/small_jump.ogg")
            self.death = mixer.Sound("./sfx/death.wav")
            self.kick = mixer.Sound("./sfx/kick.ogg")
            self.brick_bump = mixer.Sound("./sfx/brick-bump.ogg")
            self.powerup = mixer.Sound('./sfx/powerup.ogg')
            self.powerup_appear = mixer.Sound('./sfx/powerup_appears.ogg')
            self.pipe = mixer.Sound('./sfx/pipe.ogg')
        else:
            # Headless mode: dummy objects
            self.music_channel = DummyChannel()
            self.sfx_channel = DummyChannel()
            self.allowSFX = False
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

    def play_sfx(self, sfx):
        if self.enabled and self.allowSFX and sfx is not None:
            self.sfx_channel.play(sfx)

    def play_music(self, music):
        if self.enabled and music is not None:
            self.music_channel.play(music)


class DummyChannel:
    """Headless モード用のダミーチャンネル（何もしない）"""
    def play(self, sound, loops=0):
        pass
    
    def stop(self):
        pass
    
    def set_volume(self, volume):
        pass
    
    def get_busy(self):
        return False
