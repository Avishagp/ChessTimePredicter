import os
from datetime import datetime
import re
from tqdm import tqdm

openings = {}       # opening of the game
elo_dif = {}        # difference between the ranking of the two players
higher_ranker = {}  # the player that's ranked higher (w for white, b for black)
utc_time = {}       # time of game (hourly)
turns = {}          # will map between a player and all of their turns (for game ID 1 and white player: {1w, [...]})

FMT = '%H:%M:%S'


def parser_to_get_relevant_data(path):
    f = open(path, "r")
    dataset_final = open('dataset_final.txt', "w")
    lines = []
    flag = False

    for line in f:
        if ('Result' in line) or ('UTCTime' in line) or ('Elo' in line) or ('Opening' in line) or ('Diff' in line):
            lines.append(line)
        elif 'Termination' in line:
            if 'Normal' in line:
                for l in lines:
                    dataset_final.write(l)
                flag = True
            else:
                flag = False
                lines = []
        elif flag:
            dataset_final.write(line)
            if line != '\n':
                flag = False
                dataset_final.write('\n')
                lines = []


def calc_turn(start, end):
    a = datetime.strptime(start, FMT)
    b = datetime.strptime(end, FMT)
    if a > b:
        return a - b
    else:
        return b - a


def calc_all_turns(times):
    turns = []

    for i in range(len(times) - 1):
        current = times[i]
        next = times[i + 1]
        turns.append(calc_turn(current, next))

    return turns


def get_turns(line):
    all_times = re.findall(r"\d:\d\d:\d\d", line)
    white_times = []
    black_times = []

    for i, time in enumerate(all_times):
        if (i + 1) % 2:
            white_times.append(time)
        else:
            black_times.append(time)

    return calc_all_turns(white_times), calc_all_turns(black_times)


def parser(path):
    f = open(path, "r")
    i = 0
    whiteelo = 0

    for line in tqdm(f):
        if '"' in line:
            start_quote = line.index('"')
            end_quote = line.index('"', start_quote + 1)
            param = line[start_quote + 1: end_quote]

            if 'Result' in line:
                i += 1
            else:
                if 'UTCTime' in line:
                    utc_time[i] = datetime.strptime(param, FMT)
                elif 'WhiteElo' in line:
                    whiteelo = int(param)
                elif 'BlackElo' in line:
                    blackelo = int(param)
                    elo_dif[i] = abs(whiteelo - blackelo)
                    higher_ranker[i] = ('w' if whiteelo > blackelo else 'b')
                elif 'Opening' in line:
                    openings[i] = param
        elif line != '\n':
            turns[str(i) + 'w'], turns[str(i) + 'b'] = get_turns(line)

    print("finished parsing.")


def main():
    parser(r"snip_chess.txt")
    print("hi")


if __name__ == "__main__":
    main()
