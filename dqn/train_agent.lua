--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]


if not dqn then
    require "initenv"
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-framework', '', 'name of training framework')
cmd:option('-env', '', 'name of environment to use')
cmd:option('-game_path', '', 'path to environment file (ROM)')
cmd:option('-env_params', '', 'string of environment parameters')
cmd:option('-pool_frms', '',
           'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
-- cmd:option('-fix_pre_encoder', false, 'freeze weights on pre encoder')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', false,
           'saves the agent network in a separate file')
cmd:option('-prog_freq', 5*10^3, 'frequency of progress output')
cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 10^4, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')

cmd:option('-steps', 10^5, 'number of training steps to perform')
cmd:option('-eval_steps', 10^5, 'number of evaluation steps')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')
cmd:option('-global_fixweights', false, 'fix encoder weights')
cmd:option('-global_reshape', false, 'fix encoder weights')

cmd:text()

local opt = cmd:parse(arg)

global_args = {fixweights = opt.global_fixweights,
               reshape = opt.global_reshape}

--- General setup.
local game_env, game_actions, agent, opt = setup(opt)
-- agent.fix_pre_encoder = opt.fix_pre_encoder

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local result_file = assert(io.open(opt.name ..'.results.txt', 'w'), 'Failed to open result file')
result_file:write("step,mean_test_reward,mean_test_lowerbound,mean_train_lowerbound,test_mean_bonus\n")
result_file:flush()

local learn_start = agent.learn_start
local start_time = sys.clock()
local reward_counts = {}
local episode_counts = {}
local time_history = {}
local v_history = {}
local qmax_history = {}
local td_history = {}
local reward_history = {}
local step = 0
time_history[1] = 0

local total_reward
local nrewards
local nepisodes
local episode_reward

image = require 'image'
iterm = require 'iterm'

-- screen is (1 x 3 x 210 x 160)
local screen, reward, terminal = game_env:getState()
-- print("screen", screen:size())
-- screen = screen:reshape(1,3,210,160)
-- print("screen", screen:size())
-- iterm.image(screen)
--
-- screen = image.scale(screen, 84, 84, 'bilinear')
-- screen = torch.reshape(screen, 3,210,160):clone():float()
-- print(screen)
-- print("screen", screen[{1,1,1}], screen:size())
-- screen = image.lena():clone()

-- iterm.image(image.scale(screen,84,84))

-- print("screen", screen:size())

-- iterm.image(image.scale(image.lena(), 84,84, 'bilinear'))

-- screen = image.scale(screen, 84, 84, 'bilinear')
-- iterm.image(image.scale(screen, 84, 84, 'bilinear'))

local split = false

require 'optim'
require 'JustScale'


VAE = require '../inference/VAE'
require '../inference/KLDFlexCriterion'
require '../inference/KLDCriterion'
require '../inference/GaussianCriterion'
require '../inference/Sampler'


encoder = VAE.get_encoder(10)
decoder = VAE.get_decoder(10)

local input = nn.Identity()()
local enc_mean, enc_log_var = encoder(input):split(2)
local z = nn.Sampler()({enc_mean, enc_log_var})

local decoder_mean, decoder_var = decoder(z):split(2)
local vae = nn.gModule({input},{decoder_mean, decoder_var, enc_mean, enc_log_var})
gaussianCriterion = nn.GaussianCriterion()
-- KLD = nn.KLDFlexCriterion()
KLD = nn.KLDCriterion()

local image_scaler = nn.JustScale(80, 80)

vae:cuda()
gaussianCriterion:cuda()
KLD:cuda()
-- image_scaler:cuda()

local vae_params, vae_grads = vae:getParameters()


local vae_optim_config = {
    learningRate = 3e-3
}

local vae_optim_state = {}
local lowerbound = 0

-- util = {}
-- function util.oneHot(labels, n)
--    --[[
--    Assume labels is a 1D tensor of contiguous class IDs, starting at 1.
--    Turn it into a 2D tensor of size labels:size(1) x nUniqueLabels
--    This is a pretty dumb function, assumes your labels are nice.
--    ]]
--    local n = n or labels:max()
--    local nLabels = labels:size(1)
--    local out = labels.new(nLabels, n):fill(0)
--    for i=1,nLabels do
--       out[i][labels[i]] = 1.0
--    end
--    return out
-- end

-- Get/create dataset
-- if not path.exists(sys.fpath()..'/mnist') then
--   os.execute[[
--   curl https://s3.amazonaws.com/torch.data/mnist.tgz -o mnist.tgz
--   tar xvf mnist.tgz
--   rm mnist.tgz
--   ]]
-- end
--
-- local classes = {'0','1','2','3','4','5','6','7','8','9'}
-- local trainData = torch.load(sys.fpath()..'/mnist/train.t7')
-- local testData = torch.load(sys.fpath()..'/mnist/test.t7')
-- trainData.y = util.oneHot(trainData.y)
-- testData.y = util.oneHot(testData.y)

function sign(x)
    if x == 0 then return 0 end
    return x / math.abs(x)
end

print("Iteration ..", step)
local mean_lowerbound = nil
local mean_bonus = 0
while step < opt.steps do
    step = step + 1

    local vae_input = image_scaler:forward(screen:float():reshape(1, 3, 210, 160)):cuda()
    local reconstruction, reconstruction_var, mean, log_var = unpack(vae:forward(vae_input))
    reconstruction = {reconstruction, reconstruction_var}
    local err = gaussianCriterion:forward(reconstruction, vae_input)
    local KLDerr = KLD:forward(mean, log_var)
    local current_bound = - err - KLDerr

    if mean_lowerbound == nil then
        mean_lowerbound = current_bound
    end

    local bonus = sign(mean_lowerbound - current_bound)
    mean_bonus = mean_bonus * 0.95 + bonus * 0.05

    local new_reward = reward + 0.1 * bonus
    -- local new_reward = bonus
    mean_lowerbound = mean_lowerbound * 0.95 + current_bound * 0.05

    --
    local action_index = agent:perceive(new_reward, screen, terminal)
    -- local action_index = agent:perceive(reward, screen, terminal)
    --

    if step >= opt.prog_freq then
        local s, a, r, s2, term = agent.transitions:sample(agent.minibatch_size)
        -- local s, a, r, s2, term = agent.transitions:sample_raw(agent.minibatch_size)
        s = image_scaler:forward(s:float():reshape(agent.minibatch_size, 4, 3, 84, 84)[{{}, 1}]):cuda()
        -- print(s:size())

        -- raw is (4 x 3 x 210 x 160)
        -- s = s:reshape(agent.minibatch_size, 4, 3, 210, 160)[{{}, 1}]:float()
        -- s = s:reshape(agent.minibatch_size, 4, 3, 210, 160)[{{}, 1}]:float()

        -- if step % 100 == 0 then
          -- for i = 1, 10 do
          --     print('')
          --     iterm.image{s[i]}
          -- end
        -- end

        -- s = image_scaler:forward(s):cuda()
        -- print(s:size())
        -- s = image_scaler:forward(s:float():reshape(agent.minibatch_size, 4, 3, 84, 84)[{{}, 1}]):cuda()

        -- raw = torch.FloatTensor(32, 1, 32, 32)
        -- for i = 1, 32 do
        --     raw[i] = trainData.x[torch.ceil(torch.uniform(trainData.x:size(1)))]
        -- end
        -- s = image_scaler:forward(raw):cuda()

        local vae_opfunc = function(x)
            if x ~= vae_params then
                vae_params:copy(x)
            end

            --
            -- rescaled_s = torch.Tensor(s:size(1), 1, 80, 80):typeAs(s)
            -- for i = 1, s:size(1) do
            --     rescaled_s[i]:fill(image_pkg.scale(s[i], 80, 80))
            -- end
            -- s = rescaled_s
            --
            -- if s:ne(s):sum() > 0 then
            --     error("caught nan in s")
            -- end
            vae:zeroGradParameters()
            local reconstruction, reconstruction_var, mean, log_var
            reconstruction, reconstruction_var, mean, log_var = unpack(vae:forward(s))
            -- if mean:ne(mean):sum() > 0 then
            --     error("caught nan in mean")
            -- end
            -- if log_var:ne(log_var):sum() > 0 then
            --     error("caught nan in log_var")
            -- end
            -- if reconstruction:ne(reconstruction):sum() > 0 then
            --     error("caught nan in reconstruction")
            -- end
            -- if reconstruction_var:ne(reconstruction_var):sum() > 0 then
            --     error("caught nan in reconstruction_var")
            -- end

            reconstruction = {reconstruction, reconstruction_var}
            -- print("reconstruction", reconstruction)
            -- print("reconstruction", reconstruction[1][1]:size())

            if step % 1000 == 0 then
                for i = 1, 10 do
                    print('')
                    iterm.image{s[i], reconstruction[1][i]}
                    -- iterm.image{s[i]}
                    -- iterm.image{reconstruction[1][i]}
                end
            end

            -- iterm.image{reconstruction[1][1]}

            local err = gaussianCriterion:forward(reconstruction, s)
            -- if err ~= err then
            --     error("caught nan in err")
            -- end
            local df_dw = gaussianCriterion:backward(reconstruction, s)
            -- if df_dw[1]:ne(df_dw[1]):sum() > 0 then
            --     error("caught nan in df_dw[1]")
            -- end
            -- if df_dw[2]:ne(df_dw[2]):sum() > 0 then
            --     error("caught nan in df_dw[2]")
            -- end


            local KLDerr = KLD:forward(mean, log_var)
            -- if KLDerr ~= KLDerr then
            --     error("caught nan in KLDerr")
            -- end
            local dKLD_dmu, dKLD_dlog_var = unpack(KLD:backward(mean, log_var))
            -- if dKLD_dmu:ne(dKLD_dmu):sum() > 0 then
            --     error("caught nan in dKLD_dmu")
            -- end
            -- if dKLD_dlog_var:ne(dKLD_dlog_var):sum() > 0 then
            --     error("caught nan in dKLD_dlog_var")
            -- end

            local error_grads = {df_dw[1], df_dw[2], 0.05 * dKLD_dmu, 0.05 * dKLD_dlog_var}
            -- for i = 1, 4 do
            --     if error_grads[i]:ne(error_grads[i]):sum() > 0 then
            --         error("caught nan in error_grads " .. tostring(i))
            --     end
            -- end

            vae:backward(s, error_grads)

            -- if vae_grads:ne(vae_grads):sum() > 0 then
            --     error("caught nan in vae_grads")
            -- end

            local batchlowerbound = err + KLDerr
            return batchlowerbound, vae_grads
        end

        x, batchlowerbound = optim.adam(vae_opfunc, vae_params, vae_optim_config, vae_optim_state)
        lowerbound = lowerbound + batchlowerbound[1]
    end

    -- game over? get next game!
    if not terminal then
        screen, reward, terminal = game_env:step(game_actions[action_index], true)
    else
        if opt.random_starts > 0 then
            screen, reward, terminal = game_env:nextRandomGame()
        else
            screen, reward, terminal = game_env:newGame()
        end
    end


    if step % opt.prog_freq == 0 then
        assert(step==agent.numSteps, 'trainer step: ' .. step ..
                ' & agent.numSteps: ' .. agent.numSteps)
        print("Steps: ", step)
        print("VAE lower bound: ", - lowerbound / opt.prog_freq)
        print("mean bonus: ", mean_bonus)
        agent:report()
        collectgarbage()
        lowerbound = 0
    end

    if step%1000 == 0 then collectgarbage() end

    if step % opt.eval_freq == 0 and step > learn_start then
        local test_lowerbounds = {}
        local test_bonuses = {}


        screen, reward, terminal = game_env:newGame()

        total_reward = 0
        nrewards = 0
        nepisodes = 0
        episode_reward = 0

        local eval_time = sys.clock()
        for estep=1,opt.eval_steps do
            local vae_input = image_scaler:forward(screen:float():reshape(1, 3, 210, 160)):cuda()
            local reconstruction, reconstruction_var, mean, log_var = unpack(vae:forward(vae_input))
            reconstruction = {reconstruction, reconstruction_var}
            local err = gaussianCriterion:forward(reconstruction, vae_input)
            local KLDerr = KLD:forward(mean, log_var)
            local current_bound = - err - KLDerr
            table.insert(test_lowerbounds, current_bound)
            local bonus = sign(mean_lowerbound - current_bound)
            table.insert(test_bonuses, bonus)

            local action_index = agent:perceive(reward, screen, terminal, true, 0.05)

            -- Play game in test mode (episodes don't end when losing a life)
            screen, reward, terminal = game_env:step(game_actions[action_index])

            if estep%1000 == 0 then collectgarbage() end

            -- record every reward
            episode_reward = episode_reward + reward
            if reward ~= 0 then
               nrewards = nrewards + 1
            end

            if terminal then
                total_reward = total_reward + episode_reward
                episode_reward = 0
                nepisodes = nepisodes + 1
                screen, reward, terminal = game_env:nextRandomGame()
            end
        end

        eval_time = sys.clock() - eval_time
        start_time = start_time + eval_time
        agent:compute_validation_statistics(split)
        local ind = #reward_history+1
        total_reward = total_reward/math.max(1, nepisodes)

        if #reward_history == 0 or total_reward > torch.Tensor(reward_history):max() then
            agent.network:clearState()
            collectgarbage()
            agent.best_network = agent.network:clone()  -- there may be a problem here (but we just want to clone the weights right?)
        end

        if agent.v_avg then
            v_history[ind] = agent.v_avg  -- affected by validation_statistics
            td_history[ind] = agent.tderr_avg  -- affected by validation_statistics
            qmax_history[ind] = agent.q_max
        end
        print("V", v_history[ind], "TD error", td_history[ind], "Qmax", qmax_history[ind])

        reward_history[ind] = total_reward
        reward_counts[ind] = nrewards
        episode_counts[ind] = nepisodes

        time_history[ind+1] = sys.clock() - start_time

        local time_dif = time_history[ind+1] - time_history[ind]

        local training_rate = opt.actrep*opt.eval_freq/time_dif

        print(string.format(
            '\nSteps: %d (frames: %d), reward: %.2f, epsilon: %.2f, lr: %G, ' ..
            'training time: %ds, training rate: %dfps, testing time: %ds, ' ..
            'testing rate: %dfps,  num. ep.: %d,  num. rewards: %d',
            step, step*opt.actrep, total_reward, agent.ep, agent.lr, time_dif,
            training_rate, eval_time, opt.actrep*opt.eval_steps/eval_time,
            nepisodes, nrewards))
        local test_mean_lowerbound = torch.Tensor(test_lowerbounds):mean()
        local test_mean_bonus = torch.Tensor(test_bonuses):mean()
        result_file:write(step .. "," .. total_reward .. "," .. test_mean_lowerbound .. ',' .. mean_lowerbound.. ',' .. test_mean_bonus .. "\n")
        result_file:flush()
    end

    if step % opt.save_freq == 0 or step == opt.steps then
        local s, a, r, s2, term = agent.valid_s, agent.valid_a, agent.valid_r,
            agent.valid_s2, agent.valid_term
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = nil, nil, nil, nil, nil, nil, nil
        local w, dw, g, g2, delta, delta2, deltas, tmp = agent.w, agent.dw,
            agent.g, agent.g2, agent.delta, agent.delta2, agent.deltas, agent.tmp
        agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
            agent.deltas, agent.tmp = nil, nil, nil, nil, nil, nil, nil, nil

        local filename = opt.name
        if opt.save_versions > 0 then
            filename = filename .. "_" .. math.floor(step / opt.save_versions)
        end
        filename = filename

        agent.network:clearState()
        -- print(agent.best_network)
        if agent.best_network then
            agent.best_network:clearState()
        end
        collectgarbage()
        torch.save(filename .. ".t7", {agent = agent,
                                model = agent.network,
                                best_model = agent.best_network,
                                reward_history = reward_history,
                                reward_counts = reward_counts,
                                episode_counts = episode_counts,
                                time_history = time_history,
                                v_history = v_history,
                                td_history = td_history,
                                qmax_history = qmax_history,
                                arguments = opt,})
        torch.save(filename .. '.vae.t7', 
            {
                vae=vae,
                mean_lowerbound = mean_lowerbound,
            })
        if opt.saveNetworkParams then
            local nets = {network=w:clone():float()}
            torch.save(filename..'.params.t7', nets, 'ascii')
        end
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = s, a, r, s2, term
        agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
            agent.deltas, agent.tmp = w, dw, g, g2, delta, delta2, deltas, tmp
        print('Saved:', filename .. '.t7')
        io.flush()
        collectgarbage()
    end
end
