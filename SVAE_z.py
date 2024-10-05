import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, batchnorm=True,
                 nonlinearity=nn.LeakyReLU(0.2)):
        super(ConvUnit, self).__init__()
        if batchnorm is True:
            self.model = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel, stride, padding),
                nn.BatchNorm1d(out_channels), nonlinearity)
        else:
            self.model = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel, stride, padding), nonlinearity)

    def forward(self, x):
        return self.model(x)


# A block consisting of a transposed convolution, batch normalization (optional) followed by a nonlinearity (defaults to Leaky ReLU)
class ConvUnitTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, out_padding=0, batchnorm=True,
                 nonlinearity=nn.LeakyReLU(0.2)):
        super(ConvUnitTranspose, self).__init__()
        if batchnorm is True:
            self.model = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding, out_padding),
                nn.BatchNorm2d(out_channels), nonlinearity)
        else:
            self.model = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding, out_padding), nonlinearity)

    def forward(self, x):
        return self.model(x)

class LinearUnit(nn.Module):
    def __init__(self, in_features, out_features, batchnorm=True, nonlinearity=nn.LeakyReLU(0.2)):
        super(LinearUnit, self).__init__()
        if batchnorm is True and nonlinearity is not None:
            self.model = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nonlinearity)
        if batchnorm is False and nonlinearity is not None:
            self.model = nn.Sequential(
                nn.Linear(in_features, out_features),
                nonlinearity)
        if batchnorm is True and nonlinearity is None:
            self.model = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features))
        if batchnorm is False and nonlinearity is None:
            self.model = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.model(x)


class SVAE_z(nn.Module):
    def __init__(self, f_dim=256, z_dim=32, conv_dim=2048, step=256, in_size=64, hidden_dim=512, encode_dims=[48, 32],
                 decode_dims=[64, 48],
                 traj_len=8, random_sampling=True, nonlinearity=None, factorised=False, device=torch.device('cpu')):
        super(SVAE_z, self).__init__()
        self.device = device
        self.random_sampling = random_sampling
        self.f_dim = f_dim
        self.z_dim = z_dim
        self.traj_len = traj_len
        self.encode_dims = encode_dims
        self.decode_dims = decode_dims
        # self.conv_dim = conv_dim
        self.hidden_dim = hidden_dim
        self.final_conv_size = 7
        self.factorised = factorised
        self.step = step
        self.n_channels = 2  # x, y coordinates
        self.in_size = in_size
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity

        # Prior of content is a uniform Gaussian and prior of the dynamics is an LSTM
        self.z_prior_lstm = nn.LSTMCell(self.z_dim, self.hidden_dim)
        self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)
        # POSTERIOR DISTRIBUTION NETWORKS
        # -------------------------------
        # self.f_lstm = nn.LSTM(self.encode_dims[-1], self.hidden_dim, 1,
        #                       bidirectional=True, batch_first=True)
        # self.f_mean = LinearUnit(self.hidden_dim * 2, self.f_dim, False)
        # self.f_logvar = LinearUnit(self.hidden_dim * 2, self.f_dim, False)

        self.z_lstm = nn.LSTM(self.encode_dims[-1], self.hidden_dim, 1, bidirectional=True,
                              batch_first=True)
        self.z_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
        # Each timestep is for each z so no reshaping and feature mixing
        self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        Encode_Linear_list = []
        for i in range(-1, len(self.encode_dims) - 1):
            if i == -1:
                Encode_Linear_list.append(LinearUnit(self.n_channels, self.encode_dims[i + 1]))
            else:
                Encode_Linear_list.append(LinearUnit(self.encode_dims[i], self.encode_dims[i + 1]))
        self.encode_dense = nn.Sequential(*Encode_Linear_list)

        self.decode_lstm = nn.LSTM(self.z_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.decode_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
        # self.decode_dense = nn.Linear(self.hidden_dim, self.n_channels)
        Decode_Linear_list = []
        for i in range(-1, len(self.decode_dims) - 1):
            if i == -1:
                Decode_Linear_list.append(LinearUnit(self.hidden_dim, self.decode_dims[i + 1]))
            else:
                Decode_Linear_list.append(LinearUnit(self.decode_dims[i], self.decode_dims[i + 1]))
        # last fully connected layer without nolinearity in a regression manner
        Decode_Linear_list.append(LinearUnit(self.decode_dims[-1], self.n_channels, batchnorm=False, nonlinearity=None))
        self.decode_dense = nn.Sequential(*Decode_Linear_list)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    # If random sampling is true, reparametrization occurs else z_t is just set to the mean
    def sample_z(self, batch_size, random_sampling=True):
        z_out = None
        z_means = None
        z_logvars = None

        # All states are initially set to 0, especially z_0 = 0
        z_t = torch.zeros(batch_size, self.z_dim, device=self.device)
        z_mean_t = torch.zeros(batch_size, self.z_dim, device=self.device)
        z_logvar_t = torch.zeros(batch_size, self.z_dim, device=self.device)
        h_t = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=self.device)

        for _ in range(self.traj_len):
            h_t, c_t = self.z_prior_lstm(z_t, (h_t, c_t))
            z_mean_t = self.z_prior_mean(h_t)
            z_logvar_t = self.z_prior_logvar(h_t)
            z_t = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_t.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)

        return z_means, z_logvars, z_out

    def encode_trajs(self, x):
        x = x.view(-1, self.n_channels)
        x = self.encode_dense(x)
        x = x.view(-1, self.traj_len, self.encode_dims[-1])
        return x

    def decode_trajs(self, z):
        # x = self.deconv_fc(zf)
        # x = x.view(-1, self.step, self.final_conv_size, self.final_conv_size)
        # x = self.deconv(x)

        # use a simple mlp
        # !!! good for pol dataset
        # zf = zf.view(-1, zf.shape[-1])
        # x = self.decode_dense(zf)
        # x = x.view(-1, self.traj_len, self.n_channels)

        # use a recurrent decoder
        lstm_out, _ = self.decode_lstm(z)
        features, _ = self.decode_rnn(lstm_out)
        # print('features', features.shape)
        features = features.reshape(-1, features.shape[-1])
        xy = self.decode_dense(features)
        xy = xy.reshape(-1, self.traj_len, self.n_channels)
        # print('xy', xy.shape)
        return xy

    def reparameterize(self, mean, logvar, random_sampling=True):
        # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            z = mean + eps * std
            return z
        else:
            return mean

    def encode_f(self, x):
        lstm_out, _ = self.f_lstm(x)
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.traj_len - 1, 0:self.hidden_dim]
        lstm_out = torch.cat((frontal, backward), dim=1)
        mean = self.f_mean(lstm_out)
        logvar = self.f_logvar(lstm_out)
        return mean, logvar, self.reparameterize(mean, logvar, self.random_sampling)

    def encode_z(self, x):
        # The expansion is done to match the dimension of x and f, used for concatenating f to each x_t
        # f_expand = f.unsqueeze(1).expand(-1, self.traj_len, self.f_dim)
        lstm_out, _ = self.z_lstm(x)
        features, _ = self.z_rnn(lstm_out)
        mean = self.z_mean(features)
        logvar = self.z_logvar(features)
        return mean, logvar, self.reparameterize(mean, logvar, self.random_sampling)

    def sample_traj(self, batch_size, z=None):
        if z is None:
            _, _, z = self.sample_z(batch_size, random_sampling=True)
        recon_x = self.decode_trajs(z)
        return recon_x

    def forward(self, x):
        f_mean, f_logvar, f = None, None, None
        x = self.encode_trajs(x)
        # f_mean, f_logvar, f = self.encode_f(x)
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.random_sampling)
        z_mean, z_logvar, z = self.encode_z(x)
        # f_expand = f.unsqueeze(1).expand(-1, self.traj_len, self.f_dim)
        # zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decode_trajs(z)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x
