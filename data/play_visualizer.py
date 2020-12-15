from PIL import Image, ImageDraw
import conversions as conv


from play_formatter import PlayFormatter


class PlayVisualizer:
    def __init__(self, game_id, play_id, team):
        self.game_id = game_id
        self.play_id = play_id
        self.team = team

    def det_los(self):
        play__matrix = conv.unpickler(r'numpy_data\plays.npy')
        for play in play__matrix[1:, :]:
            if play[1] == self.play_id and play[0] == self.game_id:
                return int(play[19]) * 10
        raise Exception('No matching play')

    def det_bounds(self, los, routes):
        for player_id in routes.keys():
            if routes[player_id]['side'] == self.team:
                route = routes[player_id]['positions']
                first_pos = route[0]
                if int(first_pos[1] * 10) < los:
                    direction = 'left'
                else:
                    direction = 'right'
        if direction == 'left' and self.team == 'def':
            return los-80, los+220
        elif direction == 'right' and self.team == 'def':
            return los+180, los+480
        elif direction == 'left' and self.team == 'off':
            return los+100, los+400
        elif direction == 'right' and self.team == 'off':
            return los, los+300
        else:
            raise Exception('Mismatch')

    def attempt_plot(self):
        if self.not_plottable():
            pass
        else:
            self.plot_pic()

    def not_plottable(self):
        play__matrix = conv.unpickler(r'numpy_data\plays.npy')
        for play in play__matrix[1:, :]:
            if play[1] == self.play_id and play[0] == self.game_id:
                if play[19] == '':
                    return True
                else:
                    return False
        raise Exception('No matching play')

    def plot_pic(self):
        img = Image.new('L', (1600, 533), color=0)
        draw = ImageDraw.Draw(img)
        routes = PlayFormatter().format_play(self.game_id, self.play_id)
        los = self.det_los()
        draw.line((los+200, 0, los+200, 533), fill=255, width=2)
        for player_id in routes.keys():
            if routes[player_id]['side'] == self.team:
                route = routes[player_id]['positions']
                first_pos = route[0]
                draw.rectangle(((int(first_pos[1] * 10) + 190, int(first_pos[2] * 10) - 10),
                                (int(first_pos[1] * 10) + 210, int(first_pos[2] * 10) + 10)), fill=255)
                for i in range(min(len(route) - 1, 30)):
                    first_pos = route[i]
                    second_pos = route[i + 1]
                    new_first_pos = (int(first_pos[1] * 10) + 200, int(first_pos[2] * 10))
                    new_second_pos = (int(second_pos[1] * 10) + 200, int(second_pos[2] * 10))
                    draw.line((new_first_pos, new_second_pos), fill=255, width=5)
        left_bound, right_bound = self.det_bounds(los, routes)
        # img.show()
        new_img = img.crop((left_bound, 16, right_bound, 516))
        new_img = new_img.resize((37, 62), Image.ANTIALIAS)
        filename = r'play_pictures\pic-' + self.team + '-' + self.game_id + "-" + self.play_id + '.png'
        new_img.save(filename, 'PNG', quality=100)


# game_id = '2018090600'
# play_id = '75'
# team = 'def'
# vis = PlayVisualizer(game_id, play_id, team)
# vis.attempt_plot()
