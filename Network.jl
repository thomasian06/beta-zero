
include("Connect4.jl")

using Base: @kwdef

import Flux

# using Flux: relu, softmax, flatten
# using Flux: Chain, Dense, Conv, BatchNorm, SkipConnection, SamePad
using Flux
using RecursiveArrayTools
import Zygote

states_to_cpu(states) = Array{Float32}(convert(Array, VectorOfArray(states)))
add_2dim(x::Array{Float32}) = Array{Float32}(reshape(x, (size(x)..., 1, 1)))

function ResNetBlock(size, n)
    pad = size .÷ 2
    layers = Chain(
        Conv(size, n=>n, pad=pad),
        BatchNorm(n, relu),
        Conv(size, n=>n, pad=pad),
        BatchNorm(n)
    )
    return Chain(
        SkipConnection(layers, +),
        x -> relu.(x)
    )
end

mutable struct ResNet
    common
    vhead
    phead
    function ResNet()
        common = Chain(
            Conv((3,3), 1=>4, pad=SamePad()),
            BatchNorm(4, relu),
            ResNetBlock((3, 3), 4),
            ResNetBlock((3, 3), 4),
            ResNetBlock((3, 3), 4),
            ResNetBlock((3, 3), 4),
            ResNetBlock((3, 3), 4)
        )
        phead = Chain(
            Conv((1, 1), 4=>4),
            BatchNorm(4, relu),
            flatten,
            Dense(42*4, ACTION_SIZE),
            softmax
        )
        vhead = Chain(
            Conv((1, 1), 4=>1),
            BatchNorm(1, relu),
            flatten,
            Dense(42, 4, relu),
            Dense(4, 1, tanh)
        )
        new(common, vhead, phead)
    end
end

function Flux.functor(nn::ResNet)
    children = (nn.common, nn.vhead, nn.phead)
    constructor = cs -> ResNet(cs)
    return (children, constructor)
end

function forward(nn::ResNet, state)
    c = nn.common(state)
    v = nn.vhead(c)
    p = nn.phead(c)
    return(p, v)
end

(nn::ResNet)(state::Array) = forward(nn, state)

function evaluate_batch(nn::ResNet, states)
    (P, V) = nn(states_to_cpu(states))
    Pcpu = collect(eachcol(P))
    Vcpu = (V)[:]
    return collect(zip(Pcpu, Vcpu))
end