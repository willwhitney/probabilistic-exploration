--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require "nn"
require "image"

local crop = torch.class('nn.Crop', 'nn.Module')


-- function crop:__init(x1, y1, x2, y2)
--     self.x1 = x1
--     self.y1 = y1
--     self.x2 = x2
--     self.y2 = y2
-- end

function crop:__init(width, height)
    self.width = width
    self.height = height
end

function crop:forward(input)
  if input:dim() > 3 then
      input = input[1]
  end
  return image.crop(input, 'c', self.width, self.height)
end

function crop:updateOutput(input)
    return self:forward(input)
end

function crop:float()
end
