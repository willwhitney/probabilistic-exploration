require 'torch'
require 'nn'
require 'cudnn'

local VAE = {}

function VAE.get_encoder(latent_variable_size)
    SpatialConvolution = cudnn.SpatialConvolution
    SpatialBatchNormalization = cudnn.SpatialBatchNormalization

    -- 80
    local encoder = nn.Sequential()
    encoder:add(SpatialConvolution(3, 32,  3, 3,  1, 1,  1, 1))
    encoder:add(SpatialBatchNormalization(32))
    encoder:add(nn.ReLU(true))
    encoder:add(nn.SpatialMaxPooling(2, 2))

    -- 40
    encoder:add(SpatialConvolution(32, 64,  3, 3,  1, 1,  1, 1))
    encoder:add(SpatialBatchNormalization(64))
    encoder:add(nn.ReLU(true))
    encoder:add(nn.SpatialMaxPooling(2, 2))

    -- 20
    encoder:add(SpatialConvolution(64, 64,  3, 3,  1, 1,  1, 1))
    encoder:add(SpatialBatchNormalization(64))
    encoder:add(nn.ReLU(true))
    encoder:add(nn.SpatialMaxPooling(2, 2))

    -- 10x10
    encoder:add(SpatialConvolution(64, 64,  3, 3,  1, 1,  1, 1))
    encoder:add(SpatialBatchNormalization(64))
    encoder:add(nn.ReLU(true))
    encoder:add(nn.SpatialMaxPooling(2, 2))
    -- 5x5

    encoder:add(SpatialConvolution(64, 8,  3, 3,  1, 1,  1, 1))
    encoder:add(SpatialBatchNormalization(8))
    encoder:add(nn.ReLU(true))
    encoder:add(nn.Reshape(200))

    -- encoder:add(nn.Linear(400, hidden_layer_size))
    -- encoder:add(nn.BatchNormalization(hidden_layer_size))
    -- encoder:add(nn.ReLU(true))

    mean_logvar = nn.ConcatTable()
    mean_logvar:add(nn.Linear(200, latent_variable_size))
    mean_logvar:add(nn.Linear(200, latent_variable_size))

    encoder:add(mean_logvar)

    return encoder
end

function VAE.get_decoder(latent_variable_size)
    -- The Decoder
    local decoder = nn.Sequential()
    decoder:add(nn.Linear(latent_variable_size, 200))
    decoder:add(nn.BatchNormalization(200))
    decoder:add(nn.ReLU(true))

    decoder:add(nn.Reshape(8, 5, 5))

    -- 5
    decoder:add(SpatialConvolution(8, 64,  3, 3,  1, 1,  1, 1))
    decoder:add(SpatialBatchNormalization(64))
    decoder:add(nn.ReLU(true))
    decoder:add(nn.SpatialUpSamplingNearest(2))

    -- 10
    decoder:add(SpatialConvolution(64, 64,  3, 3,  1, 1,  1, 1))
    decoder:add(SpatialBatchNormalization(64))
    decoder:add(nn.ReLU(true))
    decoder:add(nn.SpatialUpSamplingNearest(2))

    --  20
    decoder:add(SpatialConvolution(64, 32,  3, 3,  1, 1,  1, 1))
    decoder:add(SpatialBatchNormalization(32))
    decoder:add(nn.ReLU(true))
    decoder:add(nn.SpatialUpSamplingNearest(2))

    -- 40
    decoder:add(SpatialConvolution(32, 32,  3, 3,  1, 1,  1, 1))
    decoder:add(SpatialBatchNormalization(32))
    decoder:add(nn.ReLU(true))
    decoder:add(nn.SpatialUpSamplingNearest(2))

    mean_logvar = nn.ConcatTable()

    mean_logvar:add(nn.Sequential()
      :add(SpatialConvolution(32, 3,  3, 3,  1, 1,  1, 1))
      :add(nn.Sigmoid()))

    mean_logvar:add(nn.Sequential()
      :add(SpatialConvolution(32, 3,  3, 3,  1, 1,  1, 1))
      -- :add(nn.Clamp(-1, -1 + 1e5)))
      :add(nn.Tanh())
      :add(nn.MulConstant(5)))

    decoder:add(mean_logvar)
    print(decoder)

    -- 80
    -- decoder:add(SpatialConvolution(64, 2,  3, 3,  1, 1,  1, 1))
    -- -- decoder:add(nn.Sigmoid())
    --
    -- mean_logvar = nn.ConcatTable()
    --
    -- mean_path = nn.Sequential()
    -- mean_path:add(nn.Select(2, 1))
    -- mean_path:add(nn.Sigmoid())
    -- mean_logvar:add(mean_path)
    --
    -- mean_logvar
    --     :add(nn.Sequential()
    --         :add(nn.Select(2, 2))
    --         :add(nn.Tanh())
    --         :add(nn.MulConstant(5)))
    --
    --         -- :add(nn.Linear())
    -- decoder:add(mean_logvar)
    return decoder
end

return VAE
