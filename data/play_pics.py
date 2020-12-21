from PIL import Image, ImageDraw
import pickle
from multiprocessing import Pool
import os


class PlayVisualizer:
    """Creates .png graphic of game play for first 3 seconds."""
    def __init__(self, play_file) -> None:
        """Initialize fields."""
        self.game_id = None
        self.play_id = None
        self.ball_snap = None
        self.los = None
        self.player_dict = None
        self.play_direction = None
        self.off_players = None
        self.def_players = None
        self.init_everything(play_file)

    def init_everything(self, play_file):
        infile = 'polished_data/play_dicts/' + play_file
        game_dict = pickle.load(open(infile, "rb"))
        self.game_id = game_dict['game_id']
        self.play_id = game_dict['play_id']
        self.ball_snap = game_dict['ball_snap']
        self.los = game_dict['los']
        self.player_dict = game_dict['player_dict']
        self.play_direction = game_dict['play_direction']
        self.off_players = game_dict['off_players']
        self.def_players = game_dict['def_players']

    def det_bounds(self, team):
        """Determines the direction to crop the image."""
        direction, los = self.play_direction, self.los
        if direction == 'left' and team == 'def':
            return los-8, los+22
        elif direction == 'right' and team == 'def':
            return los+18, los+48
        elif direction == 'left' and team == 'off':
            return los+10, los+40
        elif direction == 'right' and team == 'off':
            return los, los+30
        else:
            raise Exception('Incorrect direction or side')

    def plot_all(self, team):
        if team == 'def':
            self.plot_pic('def')
        elif team == 'off':
            self.plot_pic('off')
        elif team == 'all':
            self.plot_pic('def')
            self.plot_pic('off')
        else:
            pass

    def plot_pic(self, team) -> None:
        platoon = self.off_players if team == 'off' else self.def_players
        img = Image.new('L', (2 * 160, 2 * 54), color=0)
        draw = ImageDraw.Draw(img)
        # draw.line((self.los+20, 0, self.los+20, 54), fill=255, width=1)
        for player_id in platoon:
            route = self.player_dict[player_id]
            snap_route = route[self.ball_snap - 1:]
            x_pos, y_pos = snap_route[0][1], snap_route[0][2]
            draw.rectangle((2 * (int(x_pos) + 19), 2 * (int(y_pos) + 1),
                           2 * (int(x_pos) + 21), 2 * (int(y_pos) -1)), fill=255)
            for i in range(min(len(snap_route) - 1, 29)):
                first_pos = 2 * int((snap_route[i][1]) + 20), 2 * int(snap_route[i][2])
                second_pos = 2 * int((snap_route[i+1][1]) + 20), 2 * int(snap_route[i+1][2])
                draw.line((first_pos, second_pos), fill=255, width=2 * 1)
        left_bound, right_bound = self.det_bounds(team)
        new_img = img.crop((2 * left_bound, 0, 2 * right_bound, 2 * 54))
        if self.play_direction == 'left':
            new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
        filename = 'play_pictures/' + team + '/' + self.game_id + "-" + self.play_id + '.png'
        new_img.save(filename, 'PNG', quality=100)


def pooler(play_file):
    polisher = PlayVisualizer(play_file)
    polisher.plot_all('all')


if __name__ == "__main__":
    path, dirs, files = next(os.walk("../data/polished_data/play_dicts"))
    p = Pool(6)
    p.map(pooler, files)
    p.close()
    p.join()
