import numpy as np
import random

class TicTacToe:
    def __init__(self):        
        self.state = np.full((3,3), '-')      # the states can be '-', 'X', 'O'
        self.finish = False                   # checks if the game is finished
        print(self.state)
        print('Welcome to the TicTacToe game!')
        print('Player can be either X or O (alphabet O and not number 0)')
        print('Row and column number can be between 1-3')
        
    def reset(self):
        self.state = np.full((3,3), '-')
        print('The state is reset!')
        print(self.state)
        self.finish = False
        
    def set_position(self, player, pos_row, pos_col):
        '''
        Args:
            player (str)  : whether 'X' or 'O'
            pos_row (int) : row number to set the move (1-3)
            pos_col (int) : col number to set the move (1-3)
        '''
        if not self.finish:
            if pos_row>3 or pos_col>3:
                print('Invalid Move! The row and column numbers should be <=3')
            elif player=='X' and self.state[pos_row-1, pos_col-1]=='-':
                self.state[pos_row-1, pos_col-1] = 'X'
                print('X moved at ', (pos_row, pos_col))
            elif player=='O' and self.state[pos_row-1, pos_col-1]=='-':
                self.state[pos_row-1, pos_col-1] = 'O'
                print('O moved at ', (pos_row, pos_col))
            else:
                print('Invalid Move! Already set on this position')            
        
            if not self.finish:
                self.if_won()
                self.if_draw()
            
            print(self.state)
        else:
            print('Move not allowed because the game is finished. Please reset.')
        
    def if_won(self):
        x_won = np.array(['X', 'X', 'X'])
        o_won = np.array(['O', 'O', 'O'])
        
        # check for horizontal and vertical axes
        for i in range(3):
            if np.array_equal(self.state[i,:], x_won) or np.array_equal(self.state[:,i], x_won):
                print('X Won!!')    
                self.finish = True
            elif np.array_equal(self.state[i,:], o_won) or np.array_equal(self.state[:,i], o_won):
                print('O Won!!') 
                self.finish = True
            else:
                pass
        
        # check for diagonals
        if np.array_equal(self.state.diagonal(), x_won) or np.array_equal(np.fliplr(self.state).diagonal(), x_won):
            print('X Won!!')
            self.finish = True
            
        if np.array_equal(self.state.diagonal(), o_won) or np.array_equal(np.fliplr(self.state).diagonal(), o_won):
            print('O Won!!')
            self.finish = True
            
    def if_draw(self):
        if not np.isin('-', self.state) and not self.finish:
            self.finish=True
            print('The game is a draw! :/')        
            

def check_imm_win(state_space, player):
    '''
    checks if a players's immediate move will lead to a win and returns those coordinates
    Args:
        state_space: state of complete 3x3 matrix
        player     : the player (X or O) to check for
    ''' 
    for i in range(3):
        array_emp  = np.where(state_space[i,:]=='-')
        array_play = np.where(state_space[i,:]==player)
        if len(array_play[0])==2 and len(array_emp[0])==1:
            return [i+1,int(array_emp[0])+1]
        
        array_emp  = np.where(state_space[:,i]=='-')
        array_play = np.where(state_space[:,i]==player)
        if len(array_play[0])==2 and len(array_emp[0])==1:
            return [int(array_emp[0])+1, i+1]
        
    # check for diagonals
    array_emp   = np.where(state_space.diagonal()=='-')
    array_play  = np.where(state_space.diagonal()==player)
    if len(array_play[0])==2 and len(array_emp[0])==1:
         return [int(array_emp[0])+1, int(array_emp[0])+1] 
        
    # check for anti-diagonals
    array_emp   = np.where(np.fliplr(state_space).diagonal()=='-')
    array_play  = np.where(np.fliplr(state_space).diagonal()==player)
    if len(array_play[0])==2 and len(array_emp[0])==1:
         return [int(array_emp[0])+1, 3-int(array_emp[0])] 


def computer_player(state_space):
    '''
    impelements the logic of the computer player
    Args:
        state_space: state of complete 3x3 matrix
    '''    
    # check which states are empty to move
    empty_states = []
    for i in range(3):
        for j in range(3):
            if state_space[i,j]=='-':
                if i==0:
                    empty_states.append(j+1)
                elif i==1:
                    empty_states.append(3+j+1)
                else:
                    empty_states.append(6+j+1)
                    
    # check for computer's win
    moves = check_imm_win(state_space, 'O')
    if moves!=None:
        return moves
        
    # check for human's win
    moves = check_imm_win(state_space, 'X')
    if moves!=None:
        return moves
    
    # if above two does not return, make a valid random move
    random_state = random.choice(empty_states)
    if random_state%3==0:
        return [int(random_state/3),3]
    else:
        return [int(random_state/3)+1, random_state%3]    
            

board = TicTacToe()
flag  = 1            # to keep track of which player is playing; 0 is for O and 1 for X

while True:
    if flag==1:
        print('Turn of Human Player X')
        user_input = input("Please enter row and column number between 1-3:")
        board.set_position('X', int(user_input[0]), int(user_input[1]))
        flag = 0
    else:
        print('Turn of Computer Player O')
        out_moves = computer_player(board.state)
        board.set_position('O', out_moves[0], out_moves[1])
        flag = 1
        
    if board.finish:
        print('Game completed!')
        break




