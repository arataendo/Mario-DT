import pygame
from classes.Level import Level

class DummySound:
    def play_sfx(self, *args, **kwargs):
        pass

class DummyDashboard:
    def drawText(self, *args, **kwargs):
        pass

level = Level(None, DummySound(), DummyDashboard())
level.loadLevel('Level_custom')
print('entity_count', len(level.entityList))
for entity in level.entityList:
    if entity.__class__.__name__ in {'Goomba', 'Koopa'}:
        print(entity.__class__.__name__, entity.rect.x, entity.rect.y)
