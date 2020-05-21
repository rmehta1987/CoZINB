import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
from utils import kmeans_l1

CONST = 0


class InferenceNetwork(nn.Module):
  def __init__(self, D_vocab, D_h, inference_layers):
    super(InferenceNetwork, self).__init__()
    self.D_vocab = D_vocab
    self.D_h = D_h
    block = []
    block.append(nn.Linear(self.D_vocab, inference_layers[0]))
    block.append(nn.ReLU())
    block.append(nn.BatchNorm1d(inference_layers[0]))
    for layer in range(len(inference_layers)-1):
      block.append(nn.Linear(inference_layers[layer], inference_layers[layer+1]))
      block.append(nn.ReLU())
      block.append(nn.BatchNorm1d(inference_layers[layer+1]))
    block.append(nn.Linear(inference_layers[-1], self.D_h))
    self.block = nn.Sequential(*block)

  def forward(self, x):
    out = self.block(x)
    return out


class DecoderNetwork(nn.Module):
  def __init__(self, D_ell, D_h, decoder_layers):
    super(DecoderNetwork, self).__init__()

    block_mean = []
    block_mean.append(nn.Linear(D_ell+D_h, decoder_layers[0]))
    block_mean.append(nn.ReLU())
    block_mean.append(nn.BatchNorm1d(decoder_layers[0]))
    for layer in range(len(decoder_layers)-1):
      block_mean.append(nn.Linear(decoder_layers[layer], decoder_layers[layer+1]))
      block_mean.append(nn.ReLU())
      block_mean.append(nn.BatchNorm1d(decoder_layers[layer+1]))
    block_mean.append(nn.Linear(decoder_layers[-1], 1))
    self.block_mean = nn.Sequential(*block_mean)

    block_logvar = []
    block_logvar.append(nn.Linear(D_ell+D_h, decoder_layers[0]))
    block_logvar.append(nn.ReLU())
    block_logvar.append(nn.BatchNorm1d(decoder_layers[0]))
    for layer in range(len(decoder_layers)-1):
      block_logvar.append(nn.Linear(decoder_layers[layer], decoder_layers[layer+1]))
      block_logvar.append(nn.ReLU())
      block_logvar.append(nn.BatchNorm1d(decoder_layers[layer+1]))
    block_logvar.append(nn.Linear(decoder_layers[-1], 1))
    self.block_logvar = nn.Sequential(*block_logvar)

  def forward(self, x):
    mean = self.block_mean(x)
    logvar = self.block_logvar(x)
    return mean, torch.exp(logvar)


class CoZINB(object):
  """Correlated Zero Infalted Negative Binomial Process"""

  def __init__(self, args):
    self.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    self.lr = args.learning_rate
    self.outer_iter = int(args.outer_iteration)
    self.inner_iter = int(args.inner_iteration)
    self.dataset = args.dataset

    self.K = args.num_topics
    self.D_vocab = args.vocab_size
    self.D_ell = args.ell_size
    self.D_h = args.h_size
    self.N = args.sample_size
    self.a0 = args.a0 # hyper-parameter for h_n
    self.b0 = args.b0 # hyper-parameter for l_k
    self.p_a = args.p_a # hyper-parameter, a_{0}, for p, p ~ Beta(a_0, b_0)
    self.p_b = args.p_b # hyper-parameter, b_{0}, for p, p ~ Beta(a_0, b_0)
    self.alpha0 = args.alpha0 # hyper-parameter for pi_k and r_k
    self.eta0 = args.eta0/self.D_vocab # hyper-parameter prior for topic distribution, phi_{k}
    self.e0 = args.e0 # hyper-parameter prior for gamma_0 ~ gamma(e0,f0)
    self.f0 = args.f0 # hyper-parameter prior for gamma_0 ~ gamma(e0,f0)

    # Parameter of the base gamma process, gamma_0 ~ Gamma(e0, f0)
    self.ge0 = self.e0
    self.gf0 = 1/self.fe0
    
    # Paramater for the Chinese Restaurant table, part 1, L_jk ~ CRT(n_jk, r_jk * b_jk)
    self.primeLjk = torch.ones(self.N, self.K, device=self.device, requires_grad=True )
    
    # Parameter of the Chinese Restaurant table, part 2, L'_k ~ CRT(L_jk, gamma_0)
    self.primeLk = torch.ones(self.K, device=self.device, requires_grad=True)
      
    # 'l' the locations, that are drawn from an isotropic Gaussian with mean 0 and variance 
    # less than or equal to 1, this is required for the fitness of beta-bernoulli latent
    # feature model see Correlated Random Measures (Ranganath) Appendix Section 3
    self.ell = torch.randn(self.K, self.D_ell, requires_grad=True, device=self.device)

    # This is the parameter for the variational distribution of the words in each Factor [q(phi_k) = Dir(eta_k)]
    self.eta = self.eta0*torch.ones(self.K, self.D_vocab, device=self.device) + .01*torch.randn(self.K, self.D_vocab, device=self.device)
    
    # These are the parameter for the variational distribution of the dispersion parameter r_k [q(r_k) = Gamma(r_k1, rk_2)]
    self.rk1 = torch.ones(self.K, device=self.device)
    self.rk2 = torch.ones(self.K, device=self.device)
    
    # These are the parmeters for the variational distribution of the probability parameter p_j [q(p_j) = Beta(ap1, bp1)]
    self.ap1 = self.p_a*torch.ones(self.N, device=self.device)
    self.bp1 = self.p_a*torch.ones(self.N, device=self.device)
    
    # These are the parmeters for the variational distribution of the beta-process [q(pi_k) = Beta(ap1, bp1)]
    self.tauk1 = self.alpha0 / torch.ones(self.K, device=self.device)
    self.tauk2 = (self.alpha0*(1-1/self.K))*torch.ones(self.K, device=self.device)
    
    # These are the parmeters for the variational distribution of the Bernoulli Process [q(b_jk) = Bernoulli(nu_jk)]
    self.nu_jk = torch.rand(self.N, self.K, device=self.device)
    
    # These are the parmeters for the variational distribution of the Gamme Process [q(theta_jk) = Gamma(theta_jk1, theta_jk2)]
    self.theta_jk1 = torch.ones(N, self.K, device=self.device)
    self.theta_jk2 = torch.ones(N, self.K, device=self.device)

    # Create inference and decoder layers to generate locations
    self.inference_layers = [int(
        item) for item in args.inference_layers.split(',')]
    self.inference_network = InferenceNetwork(
        self.D_vocab, self.D_h, self.inference_layers).to(self.device)
    self.decoder_layers = [int(
        item) for item in args.decoder_layers.split(',')]
    self.decoder_network = DecoderNetwork(
        self.D_ell, self.D_h, self.decoder_layers).to(self.device)
    self.network_reg = args.network_reg
    if args.vocab_filename is not None:
      vocab_data = np.load(args.vocab_filename)
      #self.vocab = vocab_data['vocab']
      self.vocab = vocab_data[vocab_data.files[0]]
    else:
      self.vocab = None

  def fit(self, data):
    """Fit the model.
    
    Args:
      data: a structured .npz object containing:
        - data['x_idx']: A list of numpy arrays showing unique words' indices
            in each document.
        - data['x_cnt']: A list of numpy arrays showing unique words' counts
            in each document.
    """
    x_idx = torch.load('x_idx.pt')
    x_cnt = torch.load('x_cnt.pt')
    # N is the total number of documents.
    N = len(x_idx)
    M = torch.load('M.pt') # M is a list of document unique word counts.
    x_bow = torch.load('x_bow.pt')
    # pdb.set_trace()
    
    
    # These are the parmeters for the variational distribution of the latent 
    # indicator z_jm [q(z_hm)] = Multinomial(theta_jk * phi_km)]
    # factor porpotions for each document so phi[0].shape = (Factors x Num_Words_in_document)
    # this is phi in Random Function Prior Correlation Modeling (Eq. 11) 
    self.psi = [torch.randn(self.K, M_n, device=self.device) for M_n in M]
    
    # Get K_means intialization values 
    kmeans_centers = kmeans_l1(x_bow, self.K)
    
    # K means initilzation for each factor [q(phi_k) = Dir(eta_k)]
    self.eta = N/self.K*kmeans_centers+.1*torch.randn(
        self.K, self.D_vocab, device=self.device) +1+self.eta0
    
    # Initalizing l_k with kmeans centers -- the factor correlations
    self.ell = torch.mm(kmeans_centers, 
        torch.randn(self.D_vocab, self.D_ell, device=self.device))
    self.ell = self.ell/torch.norm(
        self.ell, dim=1, keepdim=True).repeat(1,self.D_ell)
    self.ell.requires_grad = True
    
    # Initalize latent count parameter for easier inference of dispersion parameter r
    # Since there is no tractable closed up date we use gradient ascent, Ljk ~ Poisson(-r ln (1-p))
    # and n_jk ~ sum_{1 to L_jk} log p where log p is the logarthmic distribution
    self.Ljk = torch.rand(self.N, self.K, device=self.device)
    self.Ljk.requires_grad = True
    
    if self.vocab is not None:
        self.display_topics()

    # fit the model
    total_perplexity = []
    for iter_all in range(self.outer_iter):
        print(iter_all)
        
        # fit psi, the latent indicator parameters
        self.fit_psi(self.eta, self.theta_jk1, self.theta_jk2)
        
        # fit variational parameters for the beta process
        
        
        # fit variational parameters for the dispersion parameter rk
        self.fit_rk(phi, M, x_idx, z_a, z_b)
            
        # fit variational parameters for the probability parameter pj
        self.fit_pj(M)
        
        
        # fit variational parameters for the gamma process
        
        
        # fit bernoulli parameters for the the bernoulli process
        self.fit_nu(M)
        
        # Fir Gamma variational parameters
        self.fit_gamma(self.theta_jk1, self.theta_jk2, x_bow, M, N, phi, x_cnt)
        
        # Fit the Dirichlet Parameter eta for the latent variable phi (this is gamma in PRME, EQ. 12)
        self.fit_eta(N, M, phi, x_idx, x_cnt, 0) 
        
        self.fit_v_ell_inference_decoder(x_bow, N, z_a, z_b, 0)
    self.display_topics()
    self.display_topic_heatmaps(x_bow, 100)

  # individual parameters
  def fit_psi(self, M, x_idx):
      '''
        This fits the variational parameter for the latent indicator z_jkm ~ Multinomial(psi_jkm) where psi_jkm = phi_km * theta_jk
        @Args:
            M = The number of mutations in sample N
            x_idx = The index of the relevant mutation in sample J
        @Returns:
            This returns the update the latent indicator variational parameter psi, shape = J x K x M_n 
            (sample j, number of factors, and proportions of the specific mutations for each factor)
      '''
      for n, M_n in enumerate(M):
          # First calculate log prob of phi
          E_ln_phi = torch.digamma(self.eta[:,x_idx[n]]) - 
          torch.digamma(torch.mm(torch.sum(self.eta, dim=1, keepdim=True), 
                                 torch.ones(1, M_n).to(self.device))) # Expectation of Dirichlet, Shape Factors x Mutations
          
          # Then calculate log prob of theta - factor proportions in sample J - shape = K x M, 
          E_ln_theta_j = torch.mm((torch.digamma(self.theta_jk1[n,:]) + torch.log(
          self.theta_jk2[n,:])).unsqueeze(1), torch.ones(1, M_n).to(self.device)) # Expectation of Gamma
          
          # Update indicators [K x M] + [K x M]
          psi_n = torch.exp(E_ln_phi+E_ln_theta_j)/torch.mm(torch.ones(self.K, 1).to(self.device), 
                                                              torch.sum(torch.exp(E_ln_phi+E_ln_theta_j), dim=0, keepdim=True))
          # Add small noise to avoid NaN/Zeros
          self.psi[n] = psi_n.data+1e-6
    
  
  def fit_pi(self, M):
    '''
        Fits the variational parameters of the beta process
        self.tauk1 = self.alpha0 / torch.ones(self.K, device=self.device)
        self.tauk2 = (self.alpha0*(1-1/self.K))*torch.ones(self.K, device=self.device)
    '''
    
    def Ep_Lk(self, M, x_cnt):
    '''
        grad_L_k = E[log p(Lk)] + E[log p(Ljk)]
        E[log p(Lk)] + E[log p(Ljk)] = self.Lk * log(E[-base_gamma]E[ln(1-p')]) + E[base_gamma]E[ln(1-p')] 
        Gammaln(self.Lk+1) + sum_{1}^{J}E[log[self.Ljk]]
        p' = sum_{1}^J[-(b_jk)ln(1-p_j)] / sum_{1}^J[alpha0 - (b_jk)ln(1-p_j)]
    '''
        # calculate E(ln(1-p')) = E[ln(alpha0/alpha0-(b_jk)ln(1-p_j))] as E[ln(alpha0) - ln(alpha0-(b_jk)ln(1-p_j))]
        # E[ln(alpha0) - ln(alpha0-(b_jk)ln(1-p_j))] = E[ln(alpha0)] - E[ln(alpha0-(b_jk)ln(1-p_j))] 
        # and then using Jensen's E[ln(alpha0)] - lnE[alpha0-(b_jk)ln(1-p_j)] = E[ln(alpha0)] - ln(E[alpha0]-E[(b_jk)ln(1-p_j)])
        # the second part is then ln(E[alpha0]-E[(b_jk)ln(1-p_j)]) = ln(E[alpha0]-E[(b_jk)]Eln(1-p_j)])
        
        E_p_prime0_left = torch.log(self.alpha0)
        # E[(b_jk)]Eln(1-p_j)]
        E_p_prime0_right = self.nu_jk * (torch.digamma(self.bp1) - torch.digamma(self.api+self.bp1)).unsqueeze(1)
        E_p_prime0_right = self.alpha0 - torch.sum(E_p_prime0_right,dim=0)
        E_p_prime = E_p_prime0_left - torch.log(E_p_prime0_right)
        E_base_gamma = self.ge0 / self.gf0
        neg_E_base_gamma= -1 * self.ge0 * self.gf0
        sum_Ljk = torch.sum(self.Ep_Ljk(M, x_cnt),dim=0)
        leftside = self.Ljk[n]*(torch.log(neg_E_base_gamma*E_p_prime)) + E_base_gamma*E_p_prime - torch.gammaln(self.Lk+1)
        
        
    def Ep_Ljk(self, M, x_cnt):
        
    '''
        grad_L_jk = E[log p(Ljk)] + E[log p(n_jk)]
        since it is difficult to calculate expectation of the PMF of a compoud poisson distribtuion we instead use the fact that
        n_jk = SUM[1(z_ji) = k] so E[log p(n_jk)] = E[log z_ji=k] = log (psi_n * diag(x_cnt[n]) (which is the mean of a multinomial)
        E[log p(Ljk)] = self.Ljk * log(E[-r]E[b]E[ln(1-p)]) + E[r]E[b]E[ln(1-p)] - (log[self.Ljk]+gammaLN(self.Ljk))
    '''
        neg_E_r_k = -1 * self.rk1 * self.rk2 # expectation of a negative gamma random variable
        E_r_k = self.rk1 / self.rk2
        E_p_j = torch.digamma(self.bp1) - torch.digamma(self.api+self.bp1)
        Ep_Ljk = torch.ones(len(M),self.K,device=self.device)
        for n, M_n in enumerate(M):
            rightside = torch.sum(torch.mm(self.psi[n], torch.diag(x_cnt[n])),dim=1)
            leftside = self.Ljk[n]*(torch.log(neg_E_r_k*E_p_j[n]*self.nu_jk[n])) + E_r_k*E_p_j[n]*self.nu_jk[n] - torch.gammaln(self.Ljk[n]+1)
            Ep_Ljk[n] = leftside + rightside

        return Ep_Ljk

    def fit_rk(self, M, x_cnt):
    '''
        Updates based on exponential family canonical, where T=(log x, x) and n=(a,-B) for a Gamma RV ~ y(a,b)
        a = E[gamma_0] - 1 + sum_{1}_{J}(E[L_jk])
        b = 1/c - sum{1}_{J}(E[b]*E[ln(1-p)]), where c = self.alpha0 in our case
    '''

        E_base_gamma = self.ge0/self.gf0
        sum_Ljk = torch.sum(self.Ep_Ljk(M, x_cnt),dim=0)
        self.rk1 = E_base_gamma + sum_Ljk


        E_ln_1_p_j = torch.digamma(self.bp1) - torch.digamma(self.ap1+self.bp1)
        self.rk2 = 1/self.alpha0 - torch.sum(self.nu_jk * E_ln_1_p_j.unsqueeze(1)),dim=0)       
        

    def fit_eta(self, N_total, M, x_idx, x_cnt, global_iter):
    '''
        This fits the variational parameter for the proportions of Mutation M in factor K, phi_km ~ Dirichlet(eta_km), shape K x M
        The eta parameter is now proportional to the number of times a mutation occurs as well as the popularity of the Factor
        since the latent indicator parameter psi ~ Multi(eta + theta) 
        @Args:
            N_total = total number of samples (need to be used for stochastic VI)
            M = The number of mutations in sample N
            psi = Latent indicator for mutations in Sample J, psi[J].shape = [K x M[J]]
            x_cnt = count of the mutations within the sample J 
    '''
        self.eta = self.eta0*torch.ones(self.K,self.D_vocab,device=self.device)
        for n, M_n in enumerate(M):  
            if x_idx[n].size == 1:
                # again, if document only has one word, then pytorch broadcast messes 
                # things up as self.eta[:,1] is of shape [50] and not [50, 1]
                self.eta[:,x_idx[n]] += (torch.mm(self.psi[n], torch.diag(x_cnt[n]))).squeeze()
            else:
                self.eta[:,x_idx[n]] += (torch.mm(self.psi[n], torch.diag(x_cnt[n])))

    def fit_theta(self, x_bow, M, N, x_cnt):
    """
        Closed-form update for theta, the variational parameter of thetas, controls the topic score matrix of the topics in a sample
    """

    # Theta = gamrnd(ZSDS + (r_k*ones(1,N)).*Z, ones(K,1)*p_i);

        E_r_k = torch.mm(torch.ones(self.N,1),(torch.digamma(self.rk1) + torch.log(self.rk2)).unsqueeze(0)) # Shape K x 1 -> Shape J x K -- dispersion parameter
        E_b_jk = self.nu_jk
        #E_b_jk = (self.nu_jk > torch.rand(self.nu_jk.shape[0], self.nu_jk.shape[1])).double().to(self.device)
        E_p_j = torch.mm((torch.digamma(self.ap1) - torch.digamma(self.api+self.bp1)).unsqueeze(1),torch.ones(1,K)) # Shape J x 1 -> Shape J x K - probability parameter for heterogenity

        for n, M_n in enumerate(M):
        self.theta_jk1[n,:] = E_r_k[n,:] * E_b_jk[n,:] + torch.sum(
            torch.mm(self.psi[n], torch.diag(x_cnt[n])), dim=1)
        self.theta_jk2[n,:] = E_p_j[n,:]

    def fit_pj(self, M):
    '''
        Closed-form of the probability parameter for the Factor intensities based on each sample, J
        allows for heterogenity within the population
    '''
        self.ap1 = self.p_a*torch.ones(self.N, device=self.device)
        for n, M_n in enumerate(M):
            self.ap1 += M_n

            self.bp1 = self.p_b*torch.ones(self.N, device=self.device)
            E_r_k = torch.mm(torch.ones(self.N,1),(torch.digamma(self.rk1) + torch.log(self.rk2)).unsqueeze(0)) # Shape K x 1 -> Shape J x K -- dispersion parameter
            E_b_jk = self.nu_jk
            self.bp1 += torch.sum(torch.mm(E_b_jk,torch.diag(E_r_k))) 

  def fit_nu(self, x_bow, M, N, x_cnt):
    '''
        Update variational parameters of the bernoulli b_jk
    '''
    
    # First calculate the transformed weights using a lower bound based on Mean Field Theory for Sigmoid Belief Networks
    # Equation 20!!!!!
    
    h = self.inference_network(x_bow)
    hl = torch.cat((h.repeat(1,self.K).view(N*self.K, self.D_h),
        self.ell.repeat(N,1)), 1)
    decoder_output1, decoder_output2 = self.decoder_network(hl)
    decoder_mu_theta = decoder_output1.view(N, self.K)
    decoder_sigma2_theta = decoder_output2.view(N, self.K)
    E_exp_neg_theta = torch.exp(-decoder_mu_theta+decoder_sigma2_theta/2) # E[f(h,l)]
    
    # the lower bound is an apporximation for E[ln sigmoid(sigmoid^-1(pi) + F(h,l))] = E[ln sigmoid(-ln pi + ln(1-pi) - F(h,l))]
    # E[ln sigmoid(-ln pi + ln(1-pi) - F(h,l))] = -E[ln [1 + exp(-ln pi + ln(1-pi) - F(h,l))]]
    # let lb = -ln pi + ln(1-pi) - F(h,l)
    # lower bound = -c*E[lb] - ln(E[exp(-c*z) + exp((1-c)z)]) ---> use jensens inequality again for bounding the second part
    # -E[lb] = E[ln pi] - E[ln(1-pi)] + E[F(h,l)]]
    # ln(E[exp(-c*z) + exp((1-c)z)]) = ln {pi_1 * exp[((c*sigma)^2)/2 + mu*c + pi_2 * exp[(((1-c)*sigma)^2)/2 + mu*(1-c)}
    # pi_1 = pi^c * pi^-c = GF(c + a)GF(B - a) / GF(a)*GF(b)  see: https://math.stackexchange.com/questions/198504/beta-distribution-find-the-general-expression-for-exr1-xs
    # pi_2 = GF(c + a - 1)GF(B + c -1) / GF(a)*GF(b)
    # where GF = Gamma function
    # E[F(h,l)]] = mu(h, l) = decoder_mu_theta
    
    # nu_jk = -E[lb] + E[r]*E[ln(1-p)] = DG(a) - DG(b) , dg = digamma
    
    lb = torch.digamma(self.tauk1) - torch.digamma(self.tauk2)
    E_lb = torch.sum(lb, decoder_mu_theta, dim=0)
    
       
    # calculate canonical update log nu_jk / 1 - nu_jk = 1 / 1 + exp(-(-E[lb] + E[r_k]E[ln(1-p_j))]
    
    E_p_j = self.ap1 / (self.ap1 + self.bp1)
    E_rk = self.rk1 / self.rk2 # expectation of a gamma random variable
    
    for n, M_n in enumerate(M):
        self.nu_jk[n,:] =  F.sigmoid(E_lb[n,:]+torch.dot(E_rk,E_p_j[n]))
        self.nu_jk[n,:] += 1e-9

  def fit_v_ell_inference_decoder(self, x_bow, N_total, z_a, z_b, global_iter):
    optimizer_v_ell_inference_decoder = optim.Adam([
        {'params': self.v},
        {'params': self.ell},
        {'params': self.psi},
        {'params': self.inference_network.parameters()},
        {'params': self.decoder_network.parameters()}
        ], lr=self.lr)
    N = N_total
    network_iter = self.inner_iter
    prev_loss = 0   
    for iter_v_ell_inference_decoder in range(network_iter):
      optimizer_v_ell_inference_decoder.zero_grad()
      h = self.inference_network(x_bow)
      hl = torch.cat((h.repeat(1,self.K).view(N*self.K, self.D_h),
          self.ell.repeat(N,1)), 1)
      decoder_output1, decoder_output2 = self.decoder_network(hl)
      decoder_mu_theta = decoder_output1.view(N, self.K)
      decoder_sigma2_theta = decoder_output2.view(N, self.K)
      decoder_sigma2_theta.data.clamp_(min=1e-6, max=100)
      ln_p_v = (self.alpha0-1)*torch.sum(torch.log(1-self.v)) + CONST
      ln_p_k = torch.cat((
          torch.log(self.v),torch.ones(1).to(self.device)), 0) + torch.cat((
          torch.zeros(1).to(self.device),torch.cumsum(
          torch.log(1-self.v), dim=0)), 0)
      E_ln_z = torch.digamma(z_a) + torch.log(z_b)
      E_ln_p_z = -N*torch.sum(torch.lgamma(
        self.beta*torch.exp(ln_p_k))) - self.beta*torch.dot(torch.exp(ln_p_k),
        torch.sum(decoder_mu_theta-E_ln_z, dim=0)) - torch.sum(
        E_ln_z) - torch.sum(z_a*z_b*torch.exp(
        -decoder_mu_theta+decoder_sigma2_theta/2))
      E_ln_p_h = -N*self.D_h/2*torch.log(
          torch.tensor(2*np.pi*self.a0).to(self.device))-1/2/self.a0*torch.sum(
          h.pow(2))
      ln_p_ell = -self.K*self.D_ell/2*torch.log(
          2*np.pi*torch.tensor(self.b0).to(self.device)) - torch.norm(
          self.ell).pow(2)/2/self.b0
      network_norm = 0
      for param in self.inference_network.parameters():
        network_norm += torch.norm(param)
      for param in self.decoder_network.parameters():
        network_norm += torch.norm(param)
      net_norm = self.network_reg*network_norm
      loss = -ln_p_v-N_total/N*E_ln_p_z-N_total/N*E_ln_p_h-ln_p_ell+net_norm
      loss.backward()
      optimizer_v_ell_inference_decoder.step()
      self.v.data.clamp_(min=1e-6, max=1-1e-6)
      print(iter_v_ell_inference_decoder, loss)
      if torch.isnan(loss.data):
        print(h,
              ln_p_v,
              ln_p_k,
              E_ln_p_z,
              E_ln_p_h,
              ln_p_ell,
              network_norm)
        raise ValueError('Nan loss!')
      if (torch.abs((prev_loss-loss)/loss) <= 1e-6 and 
          iter_v_ell_inference_decoder>=50) or (iter_v_ell_inference_decoder
          == network_iter-1):
        break
      prev_loss = loss

  def local_bound(self, x_bow, N, M, z_a, z_b, phi, x_idx, x_cnt):
    h = self.inference_network(x_bow)
    hl = torch.cat((h.repeat(1,self.K).view(N*self.K, self.D_h),
          self.ell.detach().repeat(N,1)), 1)
    decoder_output1, decoder_output2 = self.decoder_network(hl)
    decoder_mu_theta = decoder_output1.view(N, self.K)
    decoder_sigma2_theta = decoder_output2.view(N, self.K)

    ln_p_v = (self.alpha0-1)*torch.sum(torch.log(1-self.v)) + CONST
    ln_p_k = torch.cat((
        torch.log(self.v),torch.ones(1).to(self.device)), 0) + torch.cat((
        torch.zeros(1).to(self.device),torch.cumsum(
        torch.log(1-self.v), dim=0)), 0)
    E_ln_z = torch.digamma(z_a) + torch.log(z_b)
    E_ln_eta = torch.digamma(self.gamma) - torch.digamma(
        torch.sum(self.gamma, dim=1, keepdim=True).repeat(1,self.D_vocab))
    
    E_ln_p_h = -N*self.D_h/2*torch.log(
        torch.tensor(2*np.pi*self.a0).to(self.device))-1/2/self.a0*torch.sum(
        h.pow(2))
    E_ln_p_z = -N*torch.sum(torch.lgamma(
        self.beta*torch.exp(ln_p_k))) - self.beta*torch.dot(
        torch.exp(ln_p_k), torch.sum(decoder_mu_theta-E_ln_z, dim=0)) - torch.sum(
        E_ln_z) - torch.sum(z_a*z_b*torch.exp(-decoder_mu_theta+decoder_sigma2_theta/2))
    E_ln_p_c = 0
    E_ln_p_x = 0

    H_z = torch.sum(
        z_a+torch.log(z_b)+torch.lgamma(z_a)+(1-z_a)*torch.digamma(z_a))
    H_c = 0

    for n, M_n in enumerate(M):
      sum_E_z_n = torch.dot(z_a[n,:], z_b[n,:])
      E_ln_p_c += torch.dot(torch.sum(torch.mm(phi[n],torch.diag(x_cnt[n])),
          dim=1), E_ln_z[n,:]) - torch.sum(x_cnt[n])*torch.log(sum_E_z_n)
      E_ln_p_x += torch.sum(
          torch.mm(phi[n]*E_ln_eta[:,x_idx[n]], torch.diag(x_cnt[n])))
      H_c -= torch.sum(
          torch.mm(phi[n]*torch.log(phi[n]), torch.diag(x_cnt[n])))

    l_local = (E_ln_p_h+E_ln_p_z+E_ln_p_c+E_ln_p_x) + (H_z+H_c)
    if torch.isnan(l_local.data):
      print(E_ln_p_h,
            E_ln_p_z,
            E_ln_p_c,
            E_ln_p_x,
            H_z,
            H_c)
      raise ValueError('Nan loss!')
    return l_local

  def global_bound(self):
    ln_p_ell = -self.D_ell*self.K/2*np.log(2*np.pi*self.b0) - torch.norm(
        self.ell.data).pow(2)/2/self.b0
    ln_p_v = (self.alpha0-1)*torch.sum(torch.log(1-self.v)) + (self.K-1)*(
        np.log(1+self.alpha0)-np.log(1)-np.log(self.alpha0))
    E_ln_eta = torch.digamma(self.gamma) - torch.digamma(
          torch.mm(torch.sum(self.gamma, dim=1, keepdim=True), torch.ones(
          1, self.D_vocab).to(self.device)))
    E_ln_p_eta = self.K*gammaln(
        self.D_vocab*self.gamma0) - self.K*self.D_vocab*gammaln(
        self.gamma0) + (self.gamma0-1)*torch.sum(E_ln_eta)
    H_eta =  - torch.sum(torch.lgamma(torch.sum(
        self.gamma, dim=1))) + torch.sum(
        torch.lgamma(self.gamma)) - torch.sum((self.gamma-1)*E_ln_eta)
    l_global = ln_p_ell.float().to(
        self.device)+ln_p_v.data+E_ln_p_eta.float().to(self.device)+H_eta
    return l_global

  def display_topics(self, top_n_words=8, top_n_similar_topics=5):
    for k in range(self.K):
      topn_words = torch.sort(self.gamma[k,:],
                              descending=True)[1][:top_n_words]
      topk_similar_topics = torch.sort(torch.norm(
                              self.ell[k:(k+1),:].repeat(self.K,1)-self.ell,
                              dim=1))[1][1:top_n_similar_topics+1]
      print('Topic{}: Most similar to topic {}'.format(
          k, topk_similar_topics.tolist()))
      print(self.vocab[topn_words.cpu()])

  def display_topic_heatmaps(self, x_bow, nbin):
    ln_p_k = torch.cat((
        torch.log(self.v),torch.ones(1).to(self.device)), 0) + torch.cat((
        torch.zeros(1).to(self.device),torch.cumsum(
        torch.log(1-self.v), dim=0)), 0)
    h = self.inference_network(x_bow)
    N, _ = h.shape
    h_mean = torch.mean(h, dim=0, keepdim=True)
    h_centered =  h-h_mean.repeat(N,1)
    u_h, s_h, v_h = torch.svd(h_centered)
    disp_x_array = torch.linspace( -1, 1,
        steps=nbin, dtype=torch.float).unsqueeze(1).to(self.device)
    disp_y_array = torch.linspace( -1, 1,
        steps=nbin, dtype=torch.float).unsqueeze(1).to(self.device)
    disp_xy_array = torch.cat((disp_x_array.repeat(1,nbin).view(-1,1),
        disp_y_array.repeat(nbin,1)), 1)
    xy_array = torch.mm(torch.mm(disp_xy_array, torch.diag(s_h[:2])),
                        v_h[:2,:]) + h_mean.repeat(nbin*nbin,1)
    theta = torch.ones(nbin, nbin, self.K)
    fig = plt.figure(figsize=(10,8))
    for k in range(self.K):
      hl_k = torch.cat((
          xy_array, self.ell[k:k+1,:].repeat(nbin*nbin,1)), 1)
      decoder_output_k1, decoder_output_k2 = self.decoder_network(hl_k)
      theta[:,:,k] = decoder_output_k1.view(
                          nbin, nbin) + ln_p_k[k] + np.log(self.beta)
    E_Z = torch.exp(theta)
    sum_E_Z = torch.sum(E_Z, dim=2)
    for k in range(self.K):
       ax = fig.add_subplot(np.ceil(self.K/5), 5, k+1)
       ax.imshow((E_Z[:,:,k]/sum_E_Z).detach().cpu().numpy(), 
                  vmin=0, vmax=1, cmap='jet')
       top3_words = torch.sort(self.gamma[k,:], 
                               descending=True)[1][:3]

       plt.text(10, 93, str(k+1), color='white', 
                horizontalalignment='center', fontsize=16, weight='bold')
       plt.text(60, 73, self.vocab[top3_words.cpu()][0], color='white', 
                horizontalalignment='center', fontsize=14, weight='bold')
       plt.text(60, 84, self.vocab[top3_words.cpu()][1], color='white', 
                horizontalalignment='center', fontsize=14, weight='bold')
       plt.text(60, 95, self.vocab[top3_words.cpu()][2], color='white', 
                horizontalalignment='center', fontsize=14, weight='bold')
       ax.set_axis_off()
    
    plt.savefig('run2.svg')