import brian2 as b2
from brian2.units import *
import matplotlib.pyplot as plt
import numpy as np

# Helper Function
def flatten_dict(dict_0):
    """Flatten a two-level dict by gluing keys together with underscores. 
    
    TODO: Could generalize to >2 levels with recursion. 
    """
    dict_ = dict()
    for key0, dict_1 in dict_0.items():
        try:
            for key1, value in dict_1.items():
                dict_.update({'_'.join([key0, key1]): value})
        except AttributeError:
            dict_.update({key0: dict_1})
    return dict_

# Neuron and Synapse Parameters
params = dict(
    pops = dict(
        L3pyr = dict(  # L3 pyramidal neurons
            c_m = 0.5 * nF,
            v_e = 0. * mV,  
            v_i = -70. * mV,
            
            distance = 50 * umeter, #100 * umeter,
            
            # Izh params
            a = 0.03 / ms, 
            b = -2. * nS,
            c = -50. * mV,
            d = 100. * pA,
            k = 0.8 * nS/mV,
            v_r = -60. * mV, 
            v_t = -40. * mV, 
            v_thr = 30. * mV, 
            
            # noise params
            sigma_eps = 1.0e-12 * coulomb,
            tau_eps = 2.5 * ms,
            
            # synaptic params
            tau = dict(  
                AMPA = 2.0 * ms,
                NMDA = 100. * ms,  # Decay
                NMDA_opn = 2.0 * ms,  # Opening
                GABA = 10. * ms,
            ),

            # refractory period
            ref = 2. * ms,

            g = dict(
                AMPA = 0.6 * nS, # 0.25
                NMDA = 0.6 * nS, # 0.25
                GABA = 0.6 * nS, # 0.5
            ),

            # dopamine signalling
            ## e = enzyme plasticity
            ## dc = dopamine concentration over time
            taue = 1000*ms, # decay of plasticity eligibility
            taudc = 200*ms, # decay of DA conc.
            tauw = 0.7*ms, # decay of synaptic strength
            epsilon_dopa = 0.005 # DA trace step
        )
    ),
    
    syns = dict(
        L3pyr = dict(
            L3pyr = dict(
                sig = 250 * umeter,  
                syn_prob = 0.5, 
                delay_mean = 1.0 * ms,
                delay_noise_scale = 0.25 * ms,
            )
        )
    ),   

# Extra Parameters

extra = dict(
    # TODO: Scale factor for gs 
    # Not obvious .. and not the normalizing factor
    gamma_syn = 2,
    # NMDA voltage-related params
    alpha_NMDA = 0.5 / ms,
    Mg2 = 1.,
    
    # TODO: should be part of thalamus params
    g_AMPA_ext = 5 * nS,  # why different from ssc g_AMPA? different synapses?
    
    # TODO: shouldn't be in here
    rows = 1,  # Number of rows in 2d grid of cells
    cols = 1,  # Number of columns in 2d grid of cells
    rows_base = 1,  # Baseline number of rows, to normalize currents across networks sizes
    cols_base = 1,

    Apre = .01,
    Apost = -.0105,
    taupre = 20*ms,
    taupost = 20*ms,
    wmax = .01,
    wmin = -.01
    )
)

G_namespace = dict(
                G_AMPA = 10,
                G_NMDA = 1,
                G_GABA = 1/3
                )

### Equations

# threshold, reset conditions
thres = 'v > v_thr'
reset = 'v = c; u += d'

# Neuron Population Equations
pop_model = '''
    dv / dt = (k*(v - v_r)*(v - v_t) - u + sigma_eps*sqrt(2/tau_eps)*xi - I_syn) / c_m : volt
    du / dt = a*(b*(v - v_r) - u) : amp

    I_syn = I_AMPA + I_NMDA : amp
    I_AMPA = G_AMPA * g_AMPA * (v - v_e) * s_AMPA : amp
    ds_AMPA / dt = - s_AMPA / tau_AMPA : 1

    I_NMDA = G_NMDA * g_NMDA * (v - v_e) / (1 + Mg2 * exp(-0.062 * v / mV) / 3.57) * s_NMDA_tot : amp
    s_NMDA_tot : 1
    '''

# Neuron Synapse Equations
syn_model = '''
s_NMDA_tot_post = 1 * s_NMDA : 1 (summed)
ds_NMDA / dt = - s_NMDA / tau_NMDA + alpha_NMDA * x_NMDA * (1 - s_NMDA) : 1 (clock-driven)
dx_NMDA / dt = - x_NMDA / tau_NMDA_opn : 1 (clock-driven)

w_stdp: 1
dec/dt = -ec / taue : 1 (clock-driven)
ddc/dt = -dc / taudc : 1 (clock-driven)
dw/dt = ec * (dc) / tauw : 1 (clock-driven)
dapre/dt = -apre / taupre : 1 (event-driven)
dapost/dt = -apost / taupost : 1 (event-driven)
'''

syn_delay = 'delay_mean + delay_noise_scale * randn()'
syn_p = 'syn_prob * exp(-((x_pre-x_post)**2 + (y_pre-y_post)**2)/(2*sig**2))'

# Presynaptic Spike Equations
# For external inputs
spike_on_pre = '''
s_AMPA += 1
'''
# For inter neuron spiking
stdp_syn_on_pre = '''
s_AMPA += 1 + w
apre += Apre
w_stdp = clip(w_stdp + apost, wmin, wmax)
ec = clip(ec + apost, wmin, wmax)
w = clip(w, wmin, wmax)
'''

# Postsynaptic Spike Equations
da_stdp_syn_on_post='''
apost += Apost
w_stdp = clip(w_stdp + apre, wmin, wmax)
ec = clip(ec + apre, wmin, wmax)
w = clip(w, wmin, wmax)
'''

# Scale conductance values to network size
gamma_syn = params['extra']['gamma_syn']
for chan_type in params['pops']['L3pyr']['g']:
    params['pops']['L3pyr']['g'][chan_type] *= gamma_syn



##### Build Network

# create empty network object
net = b2.Network()

# Create namespaces (makes variables callable to pop and syn objects)
pop_namespace = {**flatten_dict(params['pops']['L3pyr']),
                    **params['extra'],
                    **G_namespace}

syn_namespace = {**params['syns']['L3pyr']['L3pyr'],
                    **flatten_dict(params['pops']['L3pyr']),  # TODO: don't need to include all of these
                    **params['extra']}

# Add neuron Population to net
pop = b2.NeuronGroup(
    2,
    pop_model,
    threshold = thres,
    reset = reset,
    refractory = params['pops']['L3pyr']['ref'],
    method = 'euler', 
    namespace=pop_namespace,
    name = 'L3pyr_pop')
pop.v = pop_namespace['v_r']
net.add(pop)

# Add synapse between neurons
syn = b2.Synapses(pop, 
                  pop, 
                  model=syn_model, 
                  on_pre=stdp_syn_on_pre,
                  on_post=da_stdp_syn_on_post,
                  method= 'euler',
                  namespace=syn_namespace,
                  name = 'L3pyr_syn')
syn.connect(i=0, j=1)
syn.delay = syn_delay
net.add(syn)


##### Simulation Parameters
t_trial = 3000 * ms  # Trial time
n_nrn = 2

# Spike input generators
num_inputs = 10
input_rate = 10*Hz
spike_gen = b2.PoissonGroup(num_inputs, rates=input_rate)
spike_syn = b2.Synapses(spike_gen, pop, on_pre='s_AMPA += 1')
spike_syn.connect()
# spike_syn.connect(i = 0, j = 0)
net.add(spike_gen, spike_syn)


num_inputs = 1
input_rate = 10*Hz
spike_gen_extra = b2.PoissonGroup(num_inputs, rates=input_rate)
spike_syn_extra = b2.Synapses(spike_gen_extra, pop, on_pre='s_AMPA += 1')
spike_syn_extra.connect(j=0, i=0)
net.add(spike_gen_extra, spike_syn_extra)


# num_inputs = 10
# input_rate = 10*Hz
# spike_gen = b2.PoissonGroup(num_inputs, rates=input_rate)
# spike_syn = b2.Synapses(spike_gen, pop, on_pre='s_AMPA += (1+0.01)')
# spike_syn.connect(i = 0, j = 1)
# net.add(spike_gen, spike_syn)


# Create and add monitors to net
spike_mon = b2.SpikeMonitor(pop, name = 'spike_mon')
net.add(spike_mon)

state_mon = b2.StateMonitor(pop, 'v', record = True, name = 'v_state_mon')
net.add(state_mon)

DA_mon = b2.StateMonitor(syn, ['s_NMDA','s_AMPA', 'w', 'w_stdp', 'apre', 'apost', 'ec', 'dc' ], record = True, name = 'DA_state_mon')
net.add(DA_mon)

# Generate DA input
epsilon_dopa = 0.005
on_pre = 'dc += epsilon_dopa'
DA_gen = b2.SpikeGeneratorGroup(1, 
                                indices=[0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                times=[100, 150, 180, 260, 1200, 1260, 2110, 2180, 2210]*ms)
DA_syn = b2.Synapses(DA_gen, syn, on_pre= on_pre, model = "", method='exact', name = 'DA_L3pyr_syn')
DA_syn.connect()
net.add(DA_gen, DA_syn)


def run():
    net.run(t_trial, report='stdout')

def plot():
    L31_train = net['spike_mon'].spike_trains()

    syn_i = 0
    pre_times = L31_train[0].tolist()
    post_times = L31_train[1].tolist()

    fig, axs = plt.subplots(3, figsize = (10,8))
    fig.suptitle('Dynamics of dopamine modulated STDP', fontsize=15)

    # axs[0].set_xlabel('Time (ms)')
    axs[0].set_ylabel ('Dopamine trace (DA) [a.u.]', fontsize=12)
    axs[0].plot(net['DA_state_mon'].t/ms, net['DA_state_mon'].dc[syn_i], color = 'red')

    # axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel ('Eligibility trace ($\it{e}$) [a.u.]', fontsize=12)
    axs[1].set_ylim([-0.013, 0.013])
    for t in pre_times:
        if t == pre_times[0]:
            axs[1].axvline(x = t/ms, ymin = 0.5, ymax = 1, color = 'lightsteelblue', label = 'pre spike')
        else:
            axs[1].axvline(x = t/ms, ymin = 0.5, ymax = 1, color = 'lightsteelblue')
    for t in post_times:
        if t == post_times[0]:
            axs[1].axvline(x = t/ms, ymin = 0, ymax = 0.5, color = 'peachpuff', label = 'post spike')
        else:
            axs[1].axvline(x = t/ms, ymin = 0, ymax = 0.5, color = 'peachpuff')
    axs[1].plot(net['DA_state_mon'].t/ms, net['DA_state_mon'].ec[syn_i], color = 'limegreen')
    axs[1].legend()

    axs[2].set_xlabel('Time (ms)', fontsize=12)
    axs[2].set_ylabel ('Synaptic weight ($\it{w}$) [a.u.]', fontsize=12)
    axs[2].set_ylim([-0.015, 0.015])
    for t in pre_times:
        if t == pre_times[0]:
            axs[2].axvline(x = t/ms, ymin = 0.5, ymax = 1, color = 'lightsteelblue', label = 'pre spike')
        else:
            axs[2].axvline(x = t/ms, ymin = 0.5, ymax = 1, color = 'lightsteelblue')
    for t in post_times:
        if t == post_times[0]:
            axs[2].axvline(x = t/ms, ymin = 0, ymax = 0.5, color = 'peachpuff', label = 'post spike')
        else:
            axs[2].axvline(x = t/ms, ymin = 0, ymax = 0.5, color = 'peachpuff')
    axs[2].plot(net['DA_state_mon'].t/ms, net['DA_state_mon'].w[syn_i], color = 'm')
    axs[2].legend()
    plt.savefig('figure.png', transparent=True)

    plt.figure(2, figsize = (10,2.5))
    plt.title('Dynamics of STDP', fontsize=15)
    plt.xlabel('Time [ms]', fontsize=12)
    plt.ylabel('Synaptic weight ($\it{w}$) [a.u.]', fontsize=12)
    for t in pre_times:
        if t == pre_times[0]:
            plt.axvline(x = t/ms, ymin = 0.5, ymax = 1, color = 'lightsteelblue', label = 'pre spike')
        else:
            plt.axvline(x = t/ms, ymin = 0.5, ymax = 1, color = 'lightsteelblue')
    for t in post_times:
        if t == post_times[0]:
            plt.axvline(x = t/ms, ymin = 0, ymax = 0.5, color = 'peachpuff', label = 'post spike')
        else:
            plt.axvline(x = t/ms, ymin = 0, ymax = 0.5, color = 'peachpuff')
    plt.plot(net['DA_state_mon'].t/ms, net['DA_state_mon'].w_stdp[syn_i], color = 'm')
    plt.legend()
    plt.savefig('figure.png', transparent=True)

    fig, axs = plt.subplots(2, figsize = (10,6))
    fig.tight_layout(pad=3.0)
    axs[0].set_title('Unconditioned neuron', fontsize=15)
    axs[0].set_ylabel('Membrane potential [mV]', fontsize=12)
    axs[0].plot(net['v_state_mon'].t/ms, net['v_state_mon'].v[0]/mV, color = 'mediumblue')

    axs[1].set_title('Conditioned neuron', fontsize=15)
    axs[1].set_xlabel('Time (ms)', fontsize=12)
    axs[1].set_ylabel('Membrane potential (mV)', fontsize=12)
    axs[1].plot(net['v_state_mon'].t/ms, net['v_state_mon'].v[1]/mV, color = 'darkorange')
    plt.savefig('figure.png', transparent=True)

# # Visualize pre and post synaptic traces
# taupre = taupost = 20*ms
# wmax = 0.01
# Apre = 0.01
# Apost = -Apre*taupre/taupost*1.05

# G = b2.NeuronGroup(2, 'v:1', threshold='t>(1+i)*10*ms', refractory=100*ms)

# S = b2.Synapses(G, G,
#              '''
#              dw/dt = -w/(1000*ms) : 1
#              dapre/dt = -apre/taupre : 1 (clock-driven)
#              dapost/dt = -apost/taupost : 1 (clock-driven)
#              ''',
#              on_pre='''
#              v_post += w
#              apre += Apre
#              w = clip(w+apost, 0, wmax)
#              ''',
#              on_post='''
#              apost += Apost
#              w = clip(w+apre, 0, wmax)
#              ''', method='linear')
# S.connect(i=0, j=1)
# M = b2.StateMonitor(S, ['w', 'apre', 'apost'], record=True)

# b2.run(30*ms)

# fig, axs = plt.subplots(2, figsize = (4,8))
# axs[0].plot(M.t/ms, M.apre[0], label='pre trace')
# axs[0].plot(M.t/ms, M.apost[0], label='post trace')
# axs[0].legend()

# axs[1].plot(M.t/ms, M.w[0], label='e', color = 'limegreen')
# axs[1].legend(loc='best')
# axs[1].set_xlabel('Time (ms)')


# # for firing rate plot and calc
# fig, axs = plt.subplots(2, figsize = (10,6))
# fig.tight_layout(pad=3.0)
# axs[0].set_title('Presynaptic neuron', fontsize=15)
# axs[0].set_ylabel('Membrane potential (mV)', fontsize=12)
# axs[0].plot(net['v_state_mon'].t[:30000]/ms, net['v_state_mon'].v[0][:30000]/mV, color = 'mediumblue')

# axs[1].set_title('Postsynaptic neuron', fontsize=15)
# axs[1].set_xlabel('Time (ms)', fontsize=12)
# axs[1].set_ylabel('Membrane potential (mV)', fontsize=12)
# axs[1].plot(net['v_state_mon'].t[:30000]/ms, net['v_state_mon'].v[1][:30000]/mV, color = 'darkorange')

# print(net['spike_mon'].count)