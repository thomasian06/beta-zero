
using Crayons

ACTION_SIZE = 7

global VERT_INDS = [1, 2, 3, 4]
global HORZ_INDS = [1, 7, 13, 19]
global DIAG1_INDS = [1, 8, 15, 22]
global DIAG2_INDS = [4, 9, 14, 19]
global WIN_INDS = []

# Vertical 
for i = 1:7
    push!(WIN_INDS, VERT_INDS.+6*(i-1))
    push!(WIN_INDS, VERT_INDS.+6*(i-1).+1)
    push!(WIN_INDS, VERT_INDS.+6*(i-1).+2)
end

# Horizontal
for i = 1:6
    push!(WIN_INDS, HORZ_INDS.+(i-1))
    push!(WIN_INDS, HORZ_INDS.+(i-1).+6)
    push!(WIN_INDS, HORZ_INDS.+(i-1).+12)
    push!(WIN_INDS, HORZ_INDS.+(i-1).+18)
end

# Diagonals
for i = 1:4
    push!(WIN_INDS, DIAG1_INDS.+6*(i-1))
    push!(WIN_INDS, DIAG2_INDS.+6*(i-1))
    push!(WIN_INDS, DIAG1_INDS.+6*(i-1).+1)
    push!(WIN_INDS, DIAG2_INDS.+6*(i-1).+1)
    push!(WIN_INDS, DIAG1_INDS.+6*(i-1).+2)
    push!(WIN_INDS, DIAG2_INDS.+6*(i-1).+2)
end

mutable struct Connect4
    state::Array{Float32}
    turn::Bool
    done::Bool
    actions::Array{Int32}
    action_mask::BitArray
    moves::Int32
    r::Int32
    c::Int32
    winner::Symbol
    valid::Array{Int32}

    function Connect4()
        state = zeros(Float32, (6, 7))
        turn = false
        done = false
        actions = collect(1:7)
        action_mask = BitArray(ones(7))
        moves = 0
        r = 6
        c = 7
        winner = :none
        valid = 6*ones(Int32, 7)
        new(state, turn, done, actions, action_mask, moves, r, c, winner, valid)
    end

end

function reset!(game::Connect4)
    game.state = zeros(Float32, (6, 7))
    game.turn = false
    game.done = false
    game.actions = collect(1:7)
    game.action_mask = BitArray(ones(7))
    game.moves = 0
    game.r = 6
    game.c = 7
    game.winner = :none
    game.valid = 6*ones(Int32, 7)
    return nothing
end

function isterminal(game::Connect4)
    return game.done
end

function get_actions(game::Connect4)
    return game.actions
end

function white_playing(game::Connect4)
    return !game.turn
end

function sizeof(game::Connect4)
    return (game.r, game.c)
end

# function get_action_mask(state::Array{Float32})
#     return game.valid .!= 0
# end

function get_action_mask(game::Connect4)
    return game.valid .!= 0
end

function act!(game::Connect4, action)
    r = 0
    if isterminal(game)
        return r
    end
    if game.moves == 42
        game.done = true
        return r
    end
    if !(action in get_actions(game))
        error("Attempted Illegal Action")
        return r
    end

    game.state[game.valid[action], action] = white_playing(game) ? 1.0f0 : -1.0f0
    game.valid[action] -= 1

    game.moves += 1

    if 4.0 in [abs(sum(game.state[inds])) for inds in WIN_INDS]
        game.done = true
        game.winner = white_playing(game) ? :white : :black
        r += 1
    elseif game.moves == 42
        game.done = true
        game.winner = :none
    end

    game.action_mask = get_action_mask(game)
    game.actions = (1:7)[game.action_mask]
    game.turn = !game.turn

    return r
end

function print_board(game::Connect4)
    s = game.state
    println(crayon"ffffff")
    print("Move $(game.moves+1). ")
    if !isterminal(game)
        white_playing(game) ? print("Blue to Play: ") : print("Red to Play")
    end
    println()
    for i = 1:6
        print(" ")
        for j = 1:7
            if s[i, j] == 0
                print(crayon"ffffff", ". ")
            else
                if s[i, j] == -1.
                    print(crayon"ff6c45", "O ")
                else
                    print(crayon"4594ff", "O ")
                end
            end
        end
        println()
    end
    println(crayon"48e754", " 1 2 3 4 5 6 7 ")
    println(crayon"ffffff")
end

function act_print!(game, action)
    r = act!(game, action)
    print_board(game)
end

function play_print_game(net1, net2; mcts_iters=800, n_rand_moves = 2)
    game = Connect4()
    eval = n_rand_moves == 0
    count = 0
    while !isterminal(game)
        count += 1
        a, _ = mcts(game, net1, n_iters=mcts_iters, eval=eval, train=false)
        act_print!(game, a)
        if !isterminal(game)
            a, _ = mcts(game, net2, n_iters=mcts_iters, eval=eval, train=false)
            act_print!(game, a)
        end
        if count == n_rand_moves
            eval = true
        end
    end
    if game.winner == :white 
        println("Winner: Blue")
    elseif game.winner == :black
        println("Winner: Red")
    else
        println("Draw")
    end
    return game
end

function play_mcts(net; mcts_iters=800, n_rand_moves = 2, net_side=false)
    game = Connect4()
    eval = n_rand_moves == 0
    count = 0
    if !net_side
        while !isterminal(game)
            count += 1
            a, _ = mcts(game, net, n_iters=mcts_iters, eval=eval, train=false)
            act!(game, a)
            if !isterminal(game)
                a, _ = mcts(game, n_iters=mcts_iters, eval=eval)
                act!(game, a)
            end
            if count == n_rand_moves
                eval = true
            end
        end
    else
        while !isterminal(game)
            count += 1
            a, _ = mcts(game, n_iters=mcts_iters, eval=eval)
            act!(game, a)
            if !isterminal(game)
                a, _ = mcts(game, net, n_iters=mcts_iters, eval=eval, train=false)
                act!(game, a)
            end
            if count == n_rand_moves
                eval = true
            end
        end
    end
    if game.winner == :none 
        return 0
    end
    if game.winner == :white && !net_side
        return 1
    end
    if game.winner == :black && net_side
        return 1
    end
    return -1
end