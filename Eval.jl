
using Plots
using Pkg

using BSON: @save, @load
using Dates

include("Connect4.jl")
include("MCTS.jl")
include("Network.jl")

function evaluate_mcts(net; games=200, n_mcts_iters = 100, n_rand_moves=0)
    if games % 2 != 0
        games = games + 1
    end
    wins = 0
    opp_wins = 0
    Threads.@threads for i = 1:games/2
        w = play_mcts(net, mcts_iters=n_mcts_iters, n_rand_moves=n_rand_moves, net_side=false)
        wins += w == 1 ? 1 : 0
        opp_wins += w == -1 ? 1 : 0
    end
    Threads.@threads for i = 1:games/2
        w = play_mcts(net, mcts_iters=n_mcts_iters, n_rand_moves=n_rand_moves, net_side=true)
        wins += w == 1 ? 1 : 0
        opp_wins += w == -1 ? 1 : 0
    end
    return wins/(wins+opp_wins), wins, opp_wins
end

# @load "nethist60.bson" nethist

# results = []
# nwins = []
# nopp = []
# for net in nethist
#     @time r, w, o = evaluate_mcts(net)
#     push!(results, r)
#     push!(nwins, w)
#     push!(nopp, o)
#     println("Network won $w out of $(w+o) games: $r")
# end

# @load "eval_nethist1.bson" results nwins nopp
# @load "nethist60.bson" nethist

# @save "eval_nethist1.bson" results nwins nopp

p = plot(1:61, results, label="60 Iterations")
plot!(legend=:bottomright)
xlabel!("Network Training Iteration")
ylabel!("Win Fraction against Fixed MCTS")
title!("Network Evaluation vs. Training Iteration")
display(p)