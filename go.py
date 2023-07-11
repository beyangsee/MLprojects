import numpy as np
import random
from termcolor import colored

# play: 1 (x), -1(0), 0(o) 
# state: board status
# source: where u pick the chess
# destination: where u put it


# basic move without constraint
def do_move(state, source, dest):
    state[dest] = state[source]
    state[source] = 0
    return state

# basic kill without constraint
def capture(player, state, return_captures=False):
   
    # find all enemy's chesses between my chesses
    n = len(state)
    captures = []
    for s in np.where(state == player)[0]:
        e = s + 1
        while e < n and state[e] == -player: # when my next move is within the board and it is my enemy's chess
            e += 1 # I go further
        if e >= n: # I can't go beyond the board
            break
        if state[e] == player:
            captures.append((s+1,e-1)) # captures contains n* cases that make the kill (start_indx, end_indx) 

    for s,e in captures:
        state[s:e+1] = 0 # from start_idx to end_inx, whoever on the board (1 or -1),all to '0'

    if return_captures:
        return state, captures
    else:
        return state
    
# check valid move or not - set constraints
def valid_move(player, state, source, dest, validate_suicide=True, verbose=False):
    # senerio: not moving
    if source == dest:
        if verbose:
            print('Not even moving!')
        return False
    
    # check if source position over board
    if not 0 <= source <= len(state):
        if verbose:
            print('Source %d is not within [%d,%d] !' % (source, 0, len(state)))
        return False
    
    # check if destination position over board
    if not 0 <= dest <= len(state):
        if verbose:
            print('Destination %d is not within [%d,%d] !' % (dest, 0, len(state)))
        return False

    # check if piece belongs to player
    if state[source] != player:
        if verbose:
            print('Invalid move: cannot take piece at %d !' % source)
        return False

    # check if destination is empty
    if state[dest] != 0:
        if verbose:
            print('Invalid move: cannot put piece at %d !' % dest)
        return False

    # check suicide move = after I move and kill, my enemy still can kill me without any move.
    if validate_suicide:
        new_state = state.copy()
        new_state = do_move(new_state, source, dest) # do a move
        new_state = capture(player, new_state) # player1 tries to kill, return a new board status
        new_state = capture(-player, new_state) # player2 tries to kill, return a new board status
        if new_state[dest] == 0: # if after player1 moves to the destination, destination is 0, means he/she suicide. 
            if verbose:
                print('Suicide move: cannot put piece at %d !' % dest)
            return False

    return True


# update the board after 1 round - player moves, enemy moves.
def update(player, state):
    state = capture(player, state)
    state = capture(-player, state)
    return state

# find where to pick, where to put, no constraints
def sources_and_dests(player, state, verbose=False):
    # find sources
    sources = np.where(state == player)[0]
    if len(sources) == 0: # if player has NO chess to move, won't happen.
        if verbose:
            print('Game is already over!')
        return None, None

    #find dests
    dests = np.where(state == 0)[0]
    if len(dests) == 0:
        if verbose:
            print('Board is full!') # Not gonna happen.
        return sources, None

    return sources, dests

# find valid moves
def find_valid_moves(player, state, verbose=False):
    # find sources and dests
    sources, dests = sources_and_dests(player, state, verbose=verbose)
    if dests is None:
        return None

    # uniform distribution over all valid moves (might be slower than trying to find one valid move randomly)
    valid_moves = []
    for s in sources:
        for d in dests:
            if valid_move(player, state, s, d): 
                # I have all the source and destinations, I pair them up and save the valid move to an array
                valid_moves.append((s, d))

    return valid_moves

# give a random valid move by choosing one pair
def random_move_uniform(player, state, verbose=False):
    valid_moves = find_valid_moves(player, state, verbose=verbose)

    if len(valid_moves) == 0:
        if verbose:
            print('Cannot find any valid move!')
            return None

    move = valid_moves[np.random.randint(len(valid_moves))]

    return move

 
def random_move(player, state, allow_suicide=True, verbose=False):
    # find sources and dests
    if verbose:
         print(colored('Random moving!', 'green', attrs=['bold']))
        

    sources, dests = sources_and_dests(player, state, verbose=verbose)
    if dests is None:
        return None

    # uniform distribution over all valid moves
    sources_rand = sources.copy()
    np.random.shuffle(sources_rand)
    dests_rand = dests.copy()
    np.random.shuffle(dests_rand)
    
    # source & destination shuffled and select [0]th element
    source = sources_rand[0]
    dest = dests_rand[0]

    while not valid_move(player, state, source, dest, validate_suicide = not allow_suicide):
        dests_rand = dests_rand[1:]
        if len(dests_rand) == 0:
            sources_rand = sources_rand[1:]
            if len(sources_rand) == 0:
                if verbose:
                    print('No valid move!')
                return None
            source = sources_rand[0]
            dests_rand = dests.copy()
            np.random.shuffle(dests_rand)
        dest = dests_rand[0]

    return source, dest

# random move but in a greedy way - only take the move that can capture the most pieces
def random_move_greedy(player, state,  return_scores=False, verbose=False):
    # find sources and dests
    valid_moves = find_valid_moves(player, state, verbose=verbose)

    # scores
    scores = np.zeros(len(valid_moves))
    n_other = np.sum(state == -player)
    for i, move in enumerate(valid_moves):
        new_state = do_move(state.copy(), *move) # *move = (start, end) - the return of find_valid_moves
        new_state = capture(player, new_state)
        scores[i] = n_other - np.sum(new_state == -player) # how many I can kill, i save to array-score

    # choose from best move that kill the most chess
    best = np.where(scores == np.max(scores))[0]
    move = valid_moves[np.random.choice(best)]

    if return_scores:
        return move, valid_moves, scores
    else:
        return move


_position_to_str = { i : s for i,s in enumerate('0123456789abcd') } # index to string '0123456789abcd'
_position_from_str = { v : k for k,v in _position_to_str.items() } # from string '0123456789abcd', turn to index


def position_to_str(pos):
    return _position_to_str[pos]


def position_from_str(pos):
    return _position_from_str[pos]


_state_to_str = { 1 : 'x', 0 : '_', -1 : 'o' } # state is representative number'-1','0','1'
_str_to_state = { v : k for k,v in _state_to_str.items() }


def state_to_str(state, sep=''):
    return sep.join([_state_to_str[s] for s in state])


def state_from_str(state, sep=''):
    return np.array([_str_to_state[s] for s in state.replace(sep, '')])


def move_interactive(player, state, validate= True, verbose=False): ##################### 11th May suicide is allowed during interactive play
    if verbose:
        print(colored('Contestant moving!', 'red', attrs=['bold']))
        
    valid = False
    source, dest = None, None
    while not valid:
        if verbose:
            print('Player: %r' % state_to_str([player]))
            print(state_to_str(state, sep=' '))
            print(' '.join(position_to_str(i) for i in range(len(state))))
       
        while True: 
            inp = str(input('Choose place to pick piece from:'))
            if len(inp) == 1 and inp in '0123456789abcdq':
                if inp == 'q':
                    print('Quitting the game!')
                    return None
                break
            else:
                print('Invalid input - choose again!')
                
        source = position_from_str(inp) # convert the input into a number
        
        while True:
            inp = str(input('Choose place to put piece to:'))
            if len(inp) == 1 and inp in '0123456789abcdq':
                if inp == 'q':
                    print('Quitting the game!')
                    return None
                break
            else:
                print('Invalid input - choose again!')
        dest = position_from_str(inp)
        
        if validate:
            valid = valid_move(player, state, source, dest, validate_suicide = False, verbose=verbose)
        else:
            valid = True
        if not valid and verbose:
            print('Not a valid move - Try again!')

    return source, dest



# whoever chess < 2 is the loser
def win(player, state):
    if np.sum(state == -player) < 2:
        return True
    return False


# PUT MY DEF() INTO THIS ALAK ALAK ALAK  PUT MY DEF() INTO THIS ALAK ALAK ALAK PUT MY DEF() INTO THIS ALAK ALAK ALAK PUT MY DEF() INTO THIS ALAK ALAK ALAK

class Alak(object):# it is a family of all my relative def()

    def __init__(self, state = None, last = -1, n_tiles=14):  
        self.state = None # state: array of [0,1,....0,...] there r 14 numbers.
        self.last = None # it is just a single number to tell you who is the last player -1, 1
        self.history = None # a huge list of history, i.e. this is 1 step history: (player -1 or 1, (fr index, to index),state [before I make a move]) 
        self.reset(state=state, last=last, n_tiles=n_tiles) # tiles = size of the board = 14

    @property # make it property so I can use without ()
    def player(self):
        return -self.last 
    # if we reset, the initial player will always be -(last) 
    # we set last below as constant
    
    def reset(self, state=None, last=-1, n_tiles=14): # state = None means I set it in def() below
        self.last = last 
        if state is None:
            n_pieces = max((n_tiles - 4) // 2, 1)
            state = [1] * n_pieces + [0] * (n_tiles - 2 * n_pieces) + [-1] * n_pieces # set the board
        self.state = np.array(state) 
        self.history = []
        
    # now we start real functions below ----------------------------------------------------------------------------------------------------------

    def move(self, source, dest, check=True, verbose=True):
        if check and not valid_move(self.player, self.state, source, dest, verbose=verbose): # make sure it is a valid move
            return False
        else:
            self.history.append((self.player, (source, dest), list(self.state.copy())))

            # move
            do_move(self.state, source, dest)

            # capture
            self.state = capture(self.player, self.state)
            self.state = capture(-self.player, self.state)
    
            # switch player
            self.last = self.player

            if verbose:
                print('Move %s: %s ==> %s' % (state_to_str([self.last]),
                                              state_to_str(self.history[-1][-1],sep=' '), # before capture
                                              state_to_str(self.state, sep=' '))) # after capture
                numbers = ' '.join(position_to_str(i) for i in range(len(self.state)))
                print(' '*8 + numbers + ' ' * 5 + numbers)
                print('Moved piece from %s to %s' % (position_to_str(source), position_to_str(dest)))
                
                n_pieces_before = np.sum(np.array(self.history[-1][-1])==self.player)
                n_pieces_after = np.sum(np.array(self.state)==self.player)
                score = n_pieces_before - n_pieces_after
                print('Just captured: %d ' % score )
              

            return True
        
    # because I generated random, greedy, etc several methods, let's choose.
    

        
    def move_random(self, verbose=True):
        return self.move_using(random_move, verbose=verbose)

    def move_greedy(self, verbose=True):
        return self.move_using(random_move_greedy, verbose=verbose)

    def move_interactive(self, verbose=True):
        return self.move_using(move_interactive, verbose=verbose)


# check valid  check valid check valid check valid check valid check valid check valid check valid check valid check valid check valid check valid check valid
    def valid_moves(self, verbose=True):
        return find_valid_moves(self.player, self.state, verbose=verbose)

    def valid_move(self, source, dest, verbose=True):
        valid = valid_move(self.player, self.state, source, dest, verbose=verbose)
        if verbose:
            if valid:
                print('Move is valid!')
            else:
                print('Move is invalid!')
        return valid
    
    
 # main function to play the game  main function to play the game main function to play the game main function to play the game main function to play the game
    def move_using(self, move_function, check=False, verbose=False): # what move_function I wanna use - random, greedy...i need to put real func name - long name
        move = move_function(self.player, self.state, verbose=verbose) # use the function i put in def move_using()
        if move is None: # if can't make a move(can't even find any valid move)
            if verbose:
                print('No valid moves ugh omg no valid move!') # not possible
            return False
        return self.move(*move, check=check, verbose=verbose) 
        # no contraints move = (from, to) = fr, to # f(*move) is the same f(from, to)
        # check is in the move function to check if it's valid move

    def play(self, max_steps=10, players=('random', 'random'), verbose=True): # put string or put the whole function name
        player_to_move_function = {'random' : random_move,
                           'greedy' : random_move_greedy,
                           'interactive' : move_interactive
                          }
        
        move_first, move_second = [player_to_move_function[p] if isinstance(p, str) else p for p in players]
        # if p is string, we use the function that coresponds to string: i.e. 'random' - we use - random_move
        # we define 1st, second move

        if max_steps is None:
            max_rounds = np.inf
        step = 0
        while step < max_steps:
            if verbose:
                print(colored('\nRound %d\n' % (step+1) , attrs=['bold']))
            if not self.move_using(move_first, verbose=verbose): # if can move - just keep playing, otherwise break
                break
            if win(self.last, self.state):
                break
            if not self.move_using(move_second, verbose=verbose): # if no one wins, let's keep playing
                break
            if win(self.last, self.state):
                break
            step += 1 # counting the number of rounds
        if verbose:
            print('\nPlayed game of %d rounds' % step)
            
        if win(self.last, self.state):
            if verbose:
                print('Player %s won!' % state_to_str([self.last]))
            return self.last
        else:
            if verbose:
                print('Draw - No winner!') # within maxsteps
            return 0

    # we can use outside class function in the class.
    # but if it is in the class, if we wanna use it outside, we must: 'Alak().functionname()'
    
    def state_to_str(self, state = None): 
        if state is None:
            state = self.state
        return state_to_str(state)  

    def state_from_str(self, state):
        self.state = state_from_str(state)

    def history_item_to_str(self, item):
        player, move, state = item
        src, dst = move
        return '%r : %r -> (%d, %d)' % (state_to_str([player]), state_to_str(state), src, dst)

    def history_to_str(self):
        return '\n'.join([self.history_item_to_str(item) for item in self.history])

    def __repr__(self):
        return 'Alak[%s]' % self.state_to_str()