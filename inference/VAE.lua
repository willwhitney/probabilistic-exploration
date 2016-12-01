require 'torch'
require 'nn'
require 'cudnn'

local VAE = {}

function VAE.get_encoder(latent_variable_size)
    SpatialConvolution = cudnn.SpatialConvolution
    SpatialBatchNormalization = cudnn.SpatialBatchNormalization

    -- 80
    local encoder = nn.Sequential()
    encoder:add(SpatialConvolution(1, 64,  3, 3,  1, 1,  1, 1))
    encoder:add(SpatialBatchNormalization(64))
    encoder:add(nn.ReLU(true))
    encoder:add(nn.SpatialMaxPooling(2, 2))

    -- 40
    encoder:add(SpatialConvolution(64, 128,  3, 3,  1, 1,  1, 1))
    encoder:add(SpatialBatchNormalization(128))
    encoder:add(nn.ReLU(true))
    encoder:add(nn.SpatialMaxPooling(2, 2))

    -- 20
    encoder:add(SpatialConvolution(128, 128,  3, 3,  1, 1,  1, 1))
    encoder:add(SpatialBatchNormalization(128))
    encoder:add(nn.ReLU(true))
    encoder:add(nn.SpatialMaxPooling(2, 2))

    -- 10x10
    encoder:add(SpatialConvolution(128, 128,  3, 3,  1, 1,  1, 1))
    encoder:add(SpatialBatchNormalization(128))
    encoder:add(nn.ReLU(true))
    encoder:add(nn.SpatialMaxPooling(2, 2))
    -- 5x5

    encoder:add(SpatialConvolution(128, 16,  3, 3,  1, 1,  1, 1))
    encoder:add(SpatialBatchNormalization(16))
    encoder:add(nn.ReLU(true))
    encoder:add(nn.Reshape(400))

    -- encoder:add(nn.Linear(400, hidden_layer_size))
    -- encoder:add(nn.BatchNormalization(hidden_layer_size))
    -- encoder:add(nn.ReLU(true))

    mean_logvar = nn.ConcatTable()
    mean_logvar:add(nn.Linear(400, latent_variable_size))
    mean_logvar:add(nn.Linear(400, latent_variable_size))

    encoder:add(mean_logvar)

    return encoder
end

function VAE.get_decoder(latent_variable_size)
    -- The Decoder
    local decoder = nn.Sequential()
    decoder:add(nn.Linear(latent_variable_size, 400))
    decoder:add(nn.BatchNormalization(400))
    decoder:add(nn.ReLU(true))

    decoder:add(nn.Reshape(16, 5, 5))

    -- 5
    decoder:add(SpatialConvolution(16, 128,  3, 3,  1, 1,  1, 1))
    decoder:add(SpatialBatchNormalization(128))
    decoder:add(nn.ReLU(true))
    decoder:add(nn.SpatialUpSamplingNearest(2))

    -- 10
    decoder:add(SpatialConvolution(128, 128,  3, 3,  1, 1,  1, 1))
    decoder:add(SpatialBatchNormalization(128))
    decoder:add(nn.ReLU(true))
    decoder:add(nn.SpatialUpSamplingNearest(2))

    --  20
    decoder:add(SpatialConvolution(128, 64,  3, 3,  1, 1,  1, 1))
    decoder:add(SpatialBatchNormalization(64))
    decoder:add(nn.ReLU(true))
    decoder:add(nn.SpatialUpSamplingNearest(2))

    -- 40
    decoder:add(SpatialConvolution(64, 64,  3, 3,  1, 1,  1, 1))
    decoder:add(SpatialBatchNormalization(64))
    decoder:add(nn.ReLU(true))
    decoder:add(nn.SpatialUpSamplingNearest(2))

    -- 80
    decoder:add(SpatialConvolution(64, 2,  3, 3,  1, 1,  1, 1))
    decoder:add(nn.Sigmoid())

    mean_logvar = nn.ConcatTable()
    mean_logvar:add(nn.Select(2, 1))
    mean_logvar:add(nn.Select(2, 2))
    decoder:add(mean_logvar)
    return decoder
end

return VAE
