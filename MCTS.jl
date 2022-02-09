using Flux
using Distributions

include("Connect4.jl")
include("Network.jl")


function search!(b::Connect4, net, P, Q, N, visited, c)
    if isterminal(b)
        return 0 # REWARD FOR TERMINAL
    end

    s = deepcopy(b.state)
    if !(s in visited)
        push!(visited, s)
        policy, v = net(add_2dim(s))
        policy[.~get_action_mask(b)] .= 0
        policy ./= sum(policy)
        P[s] = policy
        Q[s] = zeros(Float32, ACTION_SIZE)
        N[s] = zeros(Float32, ACTION_SIZE)
        return -v[1]
    end

    max_u, best_a = -Inf32, -1
    for a in get_actions(b)
        u = Q[s][a] + c*P[s][a]*sqrt(sum(N[s]))/(1+N[s][a])
        if u > max_u
            max_u = u
            best_a = a
        end
    end
    a = best_a

    v = act!(b, a)
    v += search!(b, net, P, Q, N, visited, c)

    Q[s][a] = (N[s][a]*Q[s][a]+v)/(N[s][a]+1)
    N[s][a] += 1
    return -v
end

function mcts(game::Connect4, net; n_iters = 1000, c = 1, train = true, α = 0.2f0, eval = false)
    S = typeof(game.state)
    N = Dict{S, Array{Float32}}()
    Q = Dict{S, Array{Float32}}()
    P = Dict{S, Array{Float32}}()
    visited = [game.state]
    N[game.state] = zeros(Float32, ACTION_SIZE)
    Q[game.state] = zeros(Float32, ACTION_SIZE)
    p, _ = net(add_2dim(game.state))
    P[game.state] = p
    P[game.state][.~get_action_mask(game)] .= 0
    if train
        dirdist = Dirichlet(sum(get_action_mask(game)), α)
        P[game.state][get_actions(game)] .*= rand(dirdist)
    end
    P[game.state] ./= sum(P[game.state])
    for i = 1:n_iters
        search!(deepcopy(game), net, P, Q, N, visited, c)
    end
    π_new = N[game.state]./sum(N[game.state])
    if eval
        return argmax(π_new), π_new
    end
    return rand(DiscreteNonParametric(get_actions(game), π_new[get_actions(game)])), Array{Float32}(π_new)
end

function search!(b::Connect4, P, Q, N, visited, c)
    if isterminal(b)
        return 0 # REWARD FOR TERMINAL
    end

    s = deepcopy(b.state)
    if !(s in visited)
        push!(visited, s)
        policy = ones(Float32, ACTION_SIZE) ./ ACTION_SIZE
        policy[.~get_action_mask(b)] .= 0
        policy ./= sum(policy)
        P[s] = policy
        Q[s] = zeros(Float32, ACTION_SIZE)
        N[s] = zeros(Float32, ACTION_SIZE)
        sign = 1
        rollout_game = deepcopy(b)
        v = 0
        sign_r = 1.0f0
        while !isterminal(rollout_game)
            v += sign_r*act!(rollout_game, rand(get_actions(rollout_game)))
            sign_r *= -1.0f0
        end
        return -v
    end

    max_u, best_a = -Inf32, -1
    for a in get_actions(b)
        u = Q[s][a] + c*P[s][a]*sqrt(sum(N[s]))/(1+N[s][a])
        if u > max_u
            max_u = u
            best_a = a
        end
    end
    a = best_a

    v = act!(b, a)
    v += search!(b, P, Q, N, visited, c)

    Q[s][a] = (N[s][a]*Q[s][a]+v)/(N[s][a]+1)
    N[s][a] += 1
    return -v
end


function mcts(game::Connect4; n_iters = 1000, c = 1, eval = false)
    S = typeof(game.state)
    N = Dict{S, Array{Float32}}()
    Q = Dict{S, Array{Float32}}()
    P = Dict{S, Array{Float32}}()
    visited = [game.state]
    N[game.state] = zeros(Float32, ACTION_SIZE)
    Q[game.state] = zeros(Float32, ACTION_SIZE)
    P[game.state] = ones(Float32, ACTION_SIZE) ./ ACTION_SIZE
    P[game.state][.~get_action_mask(game)] .= 0
    P[game.state] ./= sum(P[game.state])
    for i = 1:n_iters
        search!(deepcopy(game), P, Q, N, visited, c)
    end
    π_new = N[game.state]./sum(N[game.state])
    if eval
        return argmax(π_new), π_new
    end
    return rand(DiscreteNonParametric(get_actions(game), π_new[get_actions(game)])), Array{Float32}(π_new)
end