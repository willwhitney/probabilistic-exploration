local KLDCriterion, parent = torch.class('nn.KLDFlexCriterion', 'nn.Criterion')

local sigma_target = 70

function KLDCriterion:updateOutput(mean, log_var)
    -- Appendix B from VAE paper: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    local mean_sq = torch.pow(mean, 2)
    -- local KLDelements = log_var:clone()
    --
    -- KLDelements:exp():mul(-1)
    -- KLDelements:add(-1, mean_sq)
    -- KLDelements:add(1)
    -- KLDelements:add(log_var)
    --
    -- self.output = -0.5 * torch.sum(KLDelements)


    local term1 = log_var:clone():mul(1/2):add(- math.log(sigma_target))
    local term2 = log_var:clone():exp():mul(-1):add(-1, mean_sq):div(2 * sigma_target^2)
    self.output = term1 + term2
    self.output:add(1/2)
    self.output = self.output:sum()

    return self.output
end

function KLDCriterion:updateGradInput(mean, log_var)
	self.gradInput = {}

    self.gradInput[1] = mean:clone():div(sigma_target^2)

    -- Fix this to be nicer
    self.gradInput[2] = torch.exp(log_var):div(2 * sigma_target^2):mul(-1):add(1/2):mul(-1)

    return self.gradInput
end
