--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require "nn"
require "image"
require "pl"

local scale = torch.class('nn.JustScale', 'nn.Module')
-- print(image)

function scale:__init(height, width)
    self.height = height
    self.width = width
end

function scale:forward(x)
    -- print(x:size())
    local output = torch.Tensor(x:size(1), 1, self.width, self.height):typeAs(x)
    for i = 1, x:size(1) do
        output[i]:copy(image.scale(x[i], self.width, self.height, 'bilinear'))
    end
    return output
end

function scale:updateOutput(input)
    return self:forward(input)
end

function scale:float()
end
