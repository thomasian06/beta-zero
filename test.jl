using CUDA
using Plots
using Pkg

using BSON: @save, @load
using Dates

using IterTools: ncycle
using FillArrays

include("Connect4.jl")
include("MCTS.jl")
include("Network.jl")
include("Eval.jl")

function collect_episodes(net; n_episodes = 100, n_mcts_iters = 800, mirror=true)
    buffer = []
    count = 0
    print("Collecting Episodes: ")
    Threads.@threads for i = 1:n_episodes
        count += 1
        if count % 10 == 0
            print("i ")
        end
        game = Connect4()
        states = []
        policies = []
        actions = []
        while !isterminal(game)
            push!(states, add_2dim(game.state))
            a, policy = mcts(game, net, n_iters=n_mcts_iters)
            push!(actions, get_actions(game))
            push!(policies, policy[get_actions(game)])
            if mirror
                push!(states, add_2dim(reverse(game.state, dims=2)))
                push!(actions, reverse(8 .- get_actions(game)))
                push!(policies, reverse(policy[get_actions(game)]))
            end
            act!(game, a)
        end
        outcomes = ones(Float32, length(states))
        if game.winner == :white
            outcomes[2:2:end] .= -1.0f0
        elseif game.winner == :black
            outcomes[1:2:end] .= -1.0f0
        else
            outcomes = zeros(Float32, length(states))
        end
        if mirror
            outcomes = reduce(vcat, Fill.(outcomes, 2))
        end
        for (state, policy, action, outcome) in zip(states, policies, actions, outcomes)
            push!(buffer, (state, policy, action, outcome))
        end
    end
    return buffer
end

function random_game(n_games)
    states = []
    for i = 1:n_games
        game = Connect4()
        while !isterminal(game)
            push!(states, game.state)
            a = rand(get_actions(game))
            act!(game, a)
        end
    end
    return states
end

# net = ResNet()
# nethist = [deepcopy(net)]

@load "nethist30.bson" nethist
nethist = deepcopy(nethist[1:4])
net = deepcopy(nethist[end])

sqnorm(x) = sum(abs2, x)
function loss(s, π_new::Vector{Float32}, actions, z::Float32; c=0.002f0)
    p, v = net(s)
    p = p[actions] / sum(p[actions])
    return Float32((-sum(π_new .* log.(p)))[1] + (z-v[1])^2)
end

function loss_batch(samples)
    cost = 0.0f0
    for sample in samples
        cost += loss(sample...)
    end
    return cost / length(samples)
end

# opt = ADAM(0.2)

# nethist = [deepcopy(net)]

eval_step = 5
num_iterations = 30
episodes_per_iteration = 100
mcts_iters_per_episode = 400
# eval_games_per_iteration = 60
# mcts_iters_per_eval = 800
λ = 0.1
n_epochs = 2
batchsize = 1024

# nethist 1-21 lambda 0.2
# nethist 22-41 lambda 0.02
# nethist 42-61 lambda 0.002

@load "eval30_hist.bson" results nwins nopp
results = results[1:3]
nwins = nwins[1:3]
nopp = nopp[1:3]

# @time r, w, o = evaluate_mcts(net, games=200, n_mcts_iters=200, n_rand_moves=0)
# push!(results, r)
# push!(nwins, w)
# push!(nopp, o)
# println("Network won $w out of $(w+o) games: $r")

# r_best = r
bestnet = deepcopy(net)

# for i = 5:num_iterations
#     if i % 10 == 0
#         global λ /= 10
#     end
#     println("Running Iteration $i. ")
#     print("    ")
    
#     @time begin
#         buffer = collect_episodes(bestnet, n_episodes=episodes_per_iteration, n_mcts_iters=mcts_iters_per_episode)
#         train_loader = Flux.Data.DataLoader(buffer, batchsize = batchsize, shuffle=true)
#         for i = 1:n_epochs
#             Flux.train!(loss_batch, Flux.params(net), train_loader, ADAM(λ))
#         end
#     end
#     print("    Evaluating: ")
#     @time r, w, o = evaluate_mcts(net, games=200, n_mcts_iters=200, n_rand_moves=0)
#     println("    New network won $r fraction of the time.")
#     if r > r_best
#         push!(results, r)
#         push!(nwins, w)
#         push!(nopp, o)
#         push!(nethist, deepcopy(net))
#         bestnet = deepcopy(net)
#         @save "nethist_best.bson" nethist
#     end
# end

