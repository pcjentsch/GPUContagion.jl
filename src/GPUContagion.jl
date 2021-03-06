module GPUContagion

using RandomNumbers.Xorshifts
using CUDA
using LightGraphs
using UnPack
using Test
using GPUArrays
using FLoops
using Random123
using BenchmarkTools
@enum AgentStatus begin
    Susceptible = 1
    Infected
    Recovered
end

# const Susceptible, Infected, Recovered = 1,2,3
const RNG = Xoroshiro128Star(1)
const nodes = 10_000

function main()
    g = BitArray(adjacency_matrix(erdos_renyi(nodes,0.005)))
    u_0 = get_u_0(nodes)
    p = get_params()
    steps = 1000
    solution = vcat([u_0], [fill(Susceptible,length(u_0)) for i in 1:steps])
    sol_cu = cu.(solution)
    cu_graph = cu(g)
    @btime solve($solution,$p,$steps,$g)
    @btime solve_cuda($sol_cu,$p,$steps,$cu_graph)
    # display(solution)
    # solve_cuda(solution,p,100,g)
end
function get_u_0(nodes)
        u_0 = fill(Susceptible,nodes)
        init_indices = rand(RNG, 1:nodes, 5)
        u_0[init_indices] .= Infected #start with five infected
        return u_0
end

function get_params()
    params = (
        p = 0.005,
        recovery_rate = 0.01,
    )
    return params
end    
function solve(solution,params,steps,g)
    for t in 1:steps
        agents_step!(t,solution[t+1],solution[t],g,params)
    end
   
    return solution
end
function solve_cuda(sol_cu,params,steps,cu_graph)
    for t in 1:steps
        agents_step_gpu!(t,sol_cu[t+1],sol_cu[t],cu_graph,params)
    end
    return sol_cu
end
function agents_step!(t,u_next,u,g,params)
    @unpack p, recovery_rate  = params  
    u_next .= u#keep state the same if nothing else happens
    
    for i in 1:length(u)
        agent_status = u[i]
        if agent_status == Susceptible
            for j in 1:length(u)
                if g[i,j] == 1 && u[j] == Infected && rand(RNG) < p
                    u_next[i] = Infected
                end
            end
        elseif agent_status == Infected
            if rand(RNG) < recovery_rate
                u_next[i] = Recovered
            end
        end
    end
end

function agents_step_gpu!(t,u_next,u,graph,params)
    u_next .= u
    function kernel(state, _, (t,u_next,u,g,params,randstate))
        @unpack p, recovery_rate  = params  
        i = linear_index(state)
        if i <= length(u)
            agent_status = u[i]
            if agent_status == Susceptible
                for j in 1:length(u)
                    if g[i,j] == 1 && u[j] == Infected && GPUArrays.gpu_rand(Float64, state, randstate) < p
                        u_next[i] = Infected
                    end
                end
            elseif agent_status == Infected
                if GPUArrays.gpu_rand(Float64, state, randstate) < recovery_rate
                    u_next[i] = Recovered
                end
            end
        end
        return nothing
    end
    gpu_call(kernel, u, (t,u_next,u,graph,params, GPUArrays.default_rng(typeof(u)).state))
end


function generate_contact_vectors!(ij_dist,ji_dist,i_to_j_contacts, j_to_i_contacts)
    rand!(RNG,ij_dist,i_to_j_contacts)
    rand!(RNG,ji_dist,j_to_i_contacts)
    l_i = length(i_to_j_contacts)
    l_j = length(j_to_i_contacts)
    contacts_sums = sum(i_to_j_contacts) - sum(j_to_i_contacts)
    sample_list_length = max(l_i,l_j) #better heuristic for this based on stddev of dist?
    index_list_i = sample(RNG,1:l_i,sample_list_length)
    index_list_j = sample(RNG,1:l_j,sample_list_length)
    sample_list_i = rand(RNG,ij_dist,sample_list_length)
    sample_list_j = rand(RNG,ji_dist,sample_list_length)
    for k = 1:sample_list_length
        if (contacts_sums != 0)
            i_index = index_list_i[k]
            j_index = index_list_j[k]    
            contacts_sums +=  j_to_i_contacts[j_index] - i_to_j_contacts[i_index]

            i_to_j_contacts[i_index] = sample_list_i[k]
            j_to_i_contacts[j_index] = sample_list_j[k]    
            contacts_sums += i_to_j_contacts[i_index] -  j_to_i_contacts[j_index]
        else
            break
        end
    end
    while contacts_sums != 0
        i_index = sample(RNG,1:l_i)
        j_index = sample(RNG,1:l_j)
        contacts_sums +=  j_to_i_contacts[j_index] - i_to_j_contacts[i_index]
        i_to_j_contacts[i_index] = rand(RNG,ij_dist)
        j_to_i_contacts[j_index] = rand(RNG,ji_dist)
        contacts_sums += i_to_j_contacts[i_index] -  j_to_i_contacts[j_index]
    end
    return nothing
end
function gpu_contact_vectors!(ij_dist,ji_dist,i_to_j_contacts, j_to_i_contacts)
    rand!(RNG,ij_dist,i_to_j_contacts)
    rand!(RNG,ji_dist,j_to_i_contacts)
    l_i = length(i_to_j_contacts)
    l_j = length(j_to_i_contacts)
    contacts_sums = sum(i_to_j_contacts) - sum(j_to_i_contacts)
    sample_list_length = max(l_i,l_j) #better heuristic for this based on stddev of dist?
    index_list_i = sample(RNG,1:l_i,sample_list_length)
    index_list_j = sample(RNG,1:l_j,sample_list_length)
    sample_list_i = rand(RNG,ij_dist,sample_list_length)
    sample_list_j = rand(RNG,ji_dist,sample_list_length)
    for k = 1:sample_list_length
        if (contacts_sums != 0)
            i_index = index_list_i[k]
            j_index = index_list_j[k]    
            contacts_sums +=  j_to_i_contacts[j_index] - i_to_j_contacts[i_index]

            i_to_j_contacts[i_index] = sample_list_i[k]
            j_to_i_contacts[j_index] = sample_list_j[k]    
            contacts_sums += i_to_j_contacts[i_index] -  j_to_i_contacts[j_index]
        else
            break
        end
    end
    while contacts_sums != 0
        i_index = sample(RNG,1:l_i)
        j_index = sample(RNG,1:l_j)
        contacts_sums +=  j_to_i_contacts[j_index] - i_to_j_contacts[i_index]
        i_to_j_contacts[i_index] = rand(RNG,ij_dist)
        j_to_i_contacts[j_index] = rand(RNG,ji_dist)
        contacts_sums += i_to_j_contacts[i_index] -  j_to_i_contacts[j_index]
    end
end

function Base.show(io::IO, status::AgentStatus) 
    if status == Susceptible
        print(io, "S")
    elseif status == Infected
        print(io, "I")
    elseif status == Recovered
        print(io, "R")
    elseif status == Immunized
        print(io, "Vac")
    end
end

end
