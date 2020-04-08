import time

import numpy as np

from utils.tree_search import Node, Tree
from utils.timer_decorator import timer

class ShelfGenerator:
    def __init__(self, shelf_size=(5,5), n_shapes=4, n_colors=4):
        self.shelf_size = shelf_size
        self.n_shapes = n_shapes
        self.n_colors = n_colors

    def new_random_shelf(self, seed=None):
        ''' Return a new random shelf matrix '''
        if seed:
            np.random.seed(seed)
        # Create indices
        indices = np.random.choice(
            self.shelf_size[0]*self.shelf_size[1],
            size = self.n_shapes*self.n_colors,
            replace=False)
        # Create empty shelf
        shelf = np.zeros(self.shelf_size[0]*self.shelf_size[1])
        # Create items
        items = np.array([i+j for j in np.arange(10,(self.n_colors+1)*10,10) \
                   for i in np.arange(1,self.n_shapes+1,1)])
        # Populate shelf with items and reshape
        shelf[indices] = items
        shelf = shelf.reshape(self.shelf_size)
        shelf = np.asarray(shelf, dtype=int)
        #shelf = np.asarray(shelf, dtype=str)
        return shelf

class ShelfSolver:
    def __init__(self):
        pass
    
    def move_item(self, shelf, obj, old_position, new_position):
        new_shelf = shelf.copy()
        new_shelf[old_position] = 0
        new_shelf[new_position] = obj
        return new_shelf

    def has_been_visited(self, shelf_state, visited_states):
        return hash(str(shelf_state)) in visited_states

    def solve_task_1(self, shelf, search_type="stack"):

        # Functions only needed for task 1 ---------

        def count_colors_in_row(row):
            colors_instances = np.zeros(n_colors)
            for ob in row:
                if ob != 0:
                    color = self.get_color(ob)
                    colors_instances[color-1] += 1
            return colors_instances

        def assign_bases(shelf):
            base_ids = dict()
            for row_id, row in enumerate(shelf):
                color_counts = count_colors_in_row(row)
                colors_with_highest_count = np.argwhere(
                    color_counts == np.max(color_counts))
                if colors_with_highest_count.shape[0] == 1:
                    if colors_with_highest_count[0,0]+1 in base_ids.values():
                        base_ids[row_id] = None # if color already has base set None
                    else:
                        base_ids[row_id] = colors_with_highest_count[0,0]+1
                elif colors_with_highest_count.shape[0] > 1:
                    base_ids[row_id] = None

            # For the leftover colors assign them randomly
            leftover_row_ids = np.argwhere(np.array(list(base_ids.values()))==None)
            leftover_row_ids = [ri[0] for ri in leftover_row_ids]
            leftover_colors = [color for color in range(1, n_colors+1) if color not in list(base_ids.values())]
            assign_ids = np.random.choice(a=leftover_row_ids, size=len(leftover_colors), replace=False)
            for i in range(len(leftover_colors)):
                base_ids[assign_ids[i]] = leftover_colors[i]

            return base_ids

        def is_in_correct_base(item, base_ids, row_idx):
            return (base_ids[row_idx] == self.get_color(item))

        def is_final_state(base_ids, shelf_state):
            is_final_state = np.ones(shelf_state.shape, dtype=bool)
            for row_idx, row in enumerate(shelf_state):
                for col_idx, item in enumerate(row):
                    if item != 0:
                        is_final_state[row_idx,col_idx] = \
                           is_in_correct_base(item, base_ids, row_idx)
                    else:
                        continue
            return is_final_state.all() # all must be True

        # ------------------------------------------

        # Features of the shelf
        n_rows = shelf.shape[0]
        n_cols = shelf.shape[1]
        color_list = list(set([self.get_color(item) for row in shelf for item in row]))
        color_list.pop(0)
        n_colors = len(color_list)
        
        # Assign the correct bases
        base_ids = assign_bases(shelf)

        # Add start state
        root_node = Node(shelf)
        tree = Tree(search_type)
        tree.add(root_node)

        # Keep track of visited states
        visited_states = []
        
        # Iterate through tree
        while tree.is_not_empty():

            node = tree.get()
            if is_final_state(base_ids, node.state):
                return node.state, node.depth

            # Create children of node and add them 
            # to tree by iterating through items
            for i in range(n_rows): # iterate thru row
                for j in range(n_cols): # iterate thru col

                    item = node.state[i,j]
                    if item == 0: # if empty space
                        continue  # do nothing

                    # If item not in correct base
                    if not is_in_correct_base(item, base_ids, i):

                        # Find correct row
                        correct_row = list(base_ids.values())\
                            .index(self.get_color(item))

                        # Find empty spaces in correct row
                        empty_space_ids = np.argwhere(
                            node.state[correct_row]==0)

                        # Add all states that move item into empty
                        # space in correct row
                        for empty_space_id in empty_space_ids:
                            new_state = self.move_item(
                                node.state, item, (i,j),
                                (correct_row, empty_space_id))
                            if not self.has_been_visited(new_state, visited_states):
                                visited_states.append(hash(str(new_state)))
                                child_node = Node(new_state, parent=node)
                                tree.add(child_node)
                    else:
                        continue
            print("Length of tree: {}\r".format(len(tree)), end="")

            if len(tree) >= 20000:
                print("Queued 20k items, terminating.")
                break

        return list(np.unique(collected_depths))

    def relu(self, value):
        if value>=0:
            return value
        else:
            return 0

    def unique_per_row_score(self, state):
        score = 0
        for row in state:
            n_zeros = len([i for i in row if i==0])
            if n_zeros == 5:
                score += 20
            else:
                colors, shapes = zip(*[(self.get_color(item), self.get_shape(item)) for item in row])
                n_colors, n_shapes = len(set(colors)), len(set(shapes))
                n_color_conflicts = 5 - n_colors - self.relu(n_zeros - 1)
                n_shape_conflicts = 5 - n_shapes - self.relu(n_zeros - 1)
                row_score = n_colors - 2*n_color_conflicts + n_shapes - 2*n_shape_conflicts
                score += row_score
        return score

    def unique_per_row_and_col_score(self, state):
        score = 0
        # Calc score of rows
        for row in state:
            n_zeros = len([i for i in row if i==0])
            if n_zeros == 5:
                score += 10
            else:
                colors = [self.get_color(item) for item in row]
                n_colors = len(set(colors))
                n_color_conflicts = 5 - n_colors - self.relu(n_zeros - 1)
                score += n_colors - 2*n_color_conflicts
        # Calc score of columns
        for col in state.transpose():
            n_zeros = len([i for i in col if i==0])
            if n_zeros == 5:
                score += 10
            else:
                colors = [self.get_color(item) for item in col]
                n_colors = len(set(colors))
                n_color_conflicts = 5 - n_colors - self.relu(n_zeros - 1)
                score += n_colors - 2*n_color_conflicts

        return score

    def solve_task(self, shelf, task, verbose=0, search_type="stack", collect=False, timeout=30):
        """
        Pseudocode: Max score algorithm
        add start state to tree
        while tree not empty:
            get state
            if state is final:
                return state and depth
            current_score = calculate score of state
            # find move that maximizes score
            new list highest_scoring_moves
            for row in state
                for item in row
                    for _row in state without current row
                        for empty space in _row
                            possible_move = move item there
                            if score of possible_move >= current score
                                append this move to stack
        """
        
        if task == 2:
            calc_score_of_state = self.unique_per_row_score
        if task == 3:
            calc_score_of_state = self.unique_per_row_and_col_score
                        
        root_node = Node(shelf)
        tree = Tree(search_type)
        tree.add(root_node)
        visited_states = []
        collected_depths = []
        n_solutions_found = 0
        start_time = time.time()
        
        while tree.is_not_empty():
            if time.time()-start_time>=timeout: # exit if takes too long
                break
                
            node = tree.get()
            state = node.state.copy()
            
            current_score = calc_score_of_state(state)
            
            if verbose: 
                print("Current score: {} | Depth: {} | Length of tree: {}\r"
                      .format(current_score, node.depth, len(tree)), end="")

            for rowID, row in enumerate(state):
                for colID, item in enumerate(row):
                    for _rowID, _row in enumerate(state):
                        if _rowID == rowID:
                            continue
                        for _colID, _item in enumerate(_row):
                            if _item == 0: # if empty space
                                possible_state = self.move_item(state, item,
                                        old_position=(rowID,colID),
                                        new_position=(_rowID, _colID))
                                possible_state_score = calc_score_of_state(possible_state)
                                
                                if possible_state_score == 60 and \
                                        not self.has_been_visited(possible_state, visited_states):
                                    visited_states.append(hash(str(possible_state)))
                                    if verbose: print('\nSolution found!')
                                    if collect:
                                        collected_depths.append(node.depth+1)
                                        n_solutions_found += 1
                                        if time.time()-start_time>=timeout or n_solutions_found>=5:
                                            return min(collected_depths)
                                    else:
                                        return possible_state, node.depth+1
                                
                                elif possible_state_score > current_score and \
                                        not self.has_been_visited(possible_state, visited_states):
                                    visited_states.append(hash(str(possible_state)))
                                    child_node = Node(possible_state, parent=node)
                                    tree.add(child_node)
                                break # only one empty space needed
                            else:
                                continue
                
        try:
            return min(collected_depths)
        except ValueError:
            if verbose: print('Fatal: No solution found.')
    
    def solve_unique_per_row(self, shelf, verbose=0, search_type="stack"):
        """
        Pseudo code for this algorithm
        
        add start state to tree
        while tree not empty:
            get state
            for row in state
                if row is final
                    continue
                else
                    for item in row
                        if item has conflicts in row
                            find row where it has no conflicts and append
                            state that moves it to first empty space
                        else 
                            continue
                    
        Possibility II:
        add start state to tree
        while tree not empty:
            get state
            find item with max conflicts
            move it to position with min conflicts
            append that state
                                           
        """
        root_node = Node(shelf)
        tree = Tree(search_type)
        tree.add(root_node)
        visited_states = []
        
        while tree.is_not_empty():
            node = tree.get()
            
            if verbose: prints("\nCurrent state:\n{}".format(node.state))
            
            if self.is_final_unique_rows_state(node.state):
                print("Solution found at depth", node.depth, "\n", node.state)
                return node.state, node.depth
            
            # for each row
            for i in range(len(node.state[0])):
                if verbose: prints("= Row {} =".format(i))

                # skip it if it is in final state
                if self.row_has_all_uniques(node.state[i]):
                    print("Row is final")
                    if verbose: prints("  Row is unique, continuing …")
                    continue
                    
                else:
                    # for each item in the row
                    for j in range(len(node.state[:,0])):
                        item = node.state[i,j]
                        if verbose: prints("  Item at ({},{}): {}".format(i,j,item))

                        if self.count_item_conflicts_in_row(item, list(node.state[i].copy())):
                            if verbose: prints("  Has conflicts in row {}, finding rows that fit".format(i))

                            # find row that has no conflict
                            for ii in range(len(node.state[0])):

                                if i == ii:
                                    if verbose: prints("    Row {} is origin row".format(ii))
                                    continue

                                if not self.count_item_conflicts_in_row(item, list(node.state[ii].copy())):
                                    if verbose: prints("    Row {} has no conflicts with item {}".format(ii, item))
                                    # see if there are empty spaces in that row
                                    try: 
                                        jj = np.argwhere(
                                            node.state[ii]==0)[0,0]
                                        if verbose: prints("      Trying to move item to ({},{})".format(ii,jj))
                                    except IndexError:
                                        if verbose: prints("      Row {} has no free spaces, skipping.".format(ii))
                                        continue
                                    # move item there
                                    new_state = self.move_item(node.state, item, 
                                        old_position=(i,j), 
                                        new_position=(ii, jj))

                                    # add to tree if not been visited
                                    if not self.has_been_visited(new_state, visited_states):
                                        visited_states.append(hash(str(new_state)))
                                        child_node = Node(new_state, parent=node)
                                        tree.add(child_node)
                                        if verbose: prints("      New move:\n{}".format(new_state))
                                        #break
                                    else:
                                        if verbose: prints("      Skipping move because it has been done before.")
                                else:
                                    if verbose: prints("    Row {} has conflicts with item {}, skipping.".format(ii, item))
                        else:
                            if verbose: prints("  … has no conflicts in row {}".format(i))
                            continue
                        #break
                    #break
    
            print("Depth: {} | Length of tree: {}\r".format(node.depth, len(tree)), end="")
        
        print("Function finished without results.")
    
    def get_color(self, item):
        return item//10
    
    def get_shape(self, item):
        return item%10
    
def prints(string):
    print(string)
    time.sleep(.1)
