from matplotlib import pyplot as plt
import torch
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.colors as colors
#from scipy.stats.stats import pearsonr
import plotly.graph_objects as go
import seaborn as sns
import pandas as pd

ez = torch.load('RUN_60_6_LayerDepth/expected_z.pt')
gamma = torch.load('RUN_60_6_LayerDepth/gamma.pt')
ell = torch.load('RUN_60_6_LayerDepth/ell.pt')
phi = torch.load('/home/ludeep/Desktop/power-law/prme/RUN_60_6_LayerDepth/phi.pt')
vocab_data = np.load('patient_vocab_all.npz')
vocab = vocab_data[vocab_data.files[0]]
del vocab_data
sumez = torch.sum(ez, dim=2)
topics = ez.shape[-1]
# only read in patient barcode, first line is mutation names
mutation_df = pd.read_csv('/home/ludeep/Desktop/power-law/prme/pat_mut_mat_all.csv',encoding='utf-8',usecols=[0],header=None,skiprows=[0]) 
# skipping nan values
tcga_tss_codes = pd.read_csv('/home/ludeep/Desktop/power-law/mutTCGA/tcga_codes.csv',encoding='utf-8',skiprows=[6,201,550]) 
M = torch.load('M_all.pt')
def plotPaintbox(ez, vocab, gamma):

    # Plotting Strength via paintboxes

    
    # https://stackoverflow.com/questions/19390320/scatterplot-contours-in-matplotlib

    disp_y_array = torch.linspace( 0, 100,steps=100, dtype=torch.float)
    disp_x_array = torch.linspace( 0, 100,steps=100, dtype=torch.float)
    sumez = torch.sum(ez, dim=2)
    topics = ez.shape[-1]

    fig = plt.figure(figsize=(20., 20.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(6, 10),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )
    for k, ax in enumerate(grid):
        
        eznorm = (ez[:,:,k]/sumez).detach().cpu().numpy()
        ax.pcolormesh(disp_x_array,disp_y_array,eznorm,vmin = -1., vmax = 1., cmap='jet')
        top3_words = torch.sort(gamma[k,:], descending=True)[1][:3]

        ax.annotate(s=vocab[top3_words.cpu()][0], xy=(.1,.1), xycoords='axes fraction', color="r")
        ax.annotate(s=vocab[top3_words.cpu()][1], xy=(.1,.25), xycoords='axes fraction', color="r")
        ax.annotate(s=vocab[top3_words.cpu()][2], xy=(.1,.4), xycoords='axes fraction', color="r")
        ax.set_axis_off()

    plt.savefig('topics_3.svg')
    
def plotTopWords(gamma, vocab, numTopics=30):
    
    assert numTopics < gamma.shape[0], "numTopics shown must be less than or equal to number of total topics (gamma.shape[0])"
    
    top10_words = []
    #fig, ax = plt.subplots()
    #fig.patch.set_visible(False)
    #ax.axis('off')
    #ax.axis('tight')

    rows = []
    for k in range(0,numTopics):
        top10_words.append(list(vocab[(torch.sort(gamma[k,:], descending=True)[1][:10]).cpu()]))
        rows.append('Topic {}'.format(k))
    #ax.table(cellText=top10_words, rowLabels=rows, loc='center', fontsize=20)
    #fig.tight_layout()
    #plt.show()
    
    
    fig = go.Figure(data=[go.Table(cells=dict(values=[np.array(rows).T.tolist(), 
                                                      *(np.array(top10_words).T.tolist())], line_color='darkslategray', 
                                              fill_color='lightcyan', align='left'))])
    fig.layout['template']['data']['table'][0]['header']['fill']['color']='rgba(0,0,0,0)'
    fig.show()
    #fig.write_image("top_words.jpeg")


def display_topics(numTopics, gamma, ell, vocab, top_n_words=10, top_n_similar_topics=5):
    for k in range(numTopics):
        topn_words = torch.sort(gamma[k,:],
                                descending=True)[1][:top_n_words]
        topk_similar_topics = torch.sort(torch.norm(ell[k:(k+1),:].repeat(numTopics,1)-ell,
                                                    dim=1))[1][1:top_n_similar_topics+1]
        print('Factor{}: Most similar to factor {}'.format(
            k, topk_similar_topics.tolist()))
        print(vocab[topn_words.cpu()])
    

def display_topics_patient_type(phi,cancer_type,patient_list,tcga_tss_codes,numTopics):
    
    tss_codes = tcga_tss_codes[tcga_tss_codes['ABV'].str.contains(cancer_type)]['TSS'].tolist()
    
    # process codes
    for i in range(len(tss_codes)):
        if len(tss_codes[i]) == 1:
            tss_codes[i] = '-0' + tss_codes[i] + '-'
        else:
            tss_codes[i] = '-' + tss_codes[i] + '-'
    
    # contains the index of barcodes that match with the patient sample of the specific cancer type
    index_of_barcodes = patient_list[patient_list[0].str.contains('|'.join(tss_codes))].index.tolist()
    phi2 = np.asarray(phi)[index_of_barcodes] # is of number of patients matching cancer type
    best_topic_in_patient = []
    best_topic_all_patient = np.zeros((numTopics,1))
    for k in range(len(phi2)):
        phi3 = phi2[k].detach().cpu() # shape num_of_topics x number of mutations in patient k
        temp = torch.sort(phi3,dim=0,descending=True)[1].numpy()
        temp_unique, temp_counts = np.unique(temp[0],return_counts=True)
        for i, unq in enumerate(temp_unique):
            best_topic_all_patient[unq] += temp_counts[i]

        #len_of_phi3 = phi3.shape[-1]
        #for j in range(len_of_phi3):
        #     # get the best topic for each mutation
        #    best_topic_in_patient.append(torch.sort(phi3[j],dim=0,descending=True)[1][0].numpy())
    df2 = pd.DataFrame(np.log10(best_topic_all_patient),index=np.arange(0,60),columns=[cancer_type])
    plt.figure(figsize=(20,20))
    sns.barplot(x=np.arange(0,60),y=cancer_type,data=df2) 
    #plt.savefig('{}_factors.png'.format(cancer_type)) 

    return df2
    
    
def norm_topics(topTopics, numTopics, gamma, ell, vocab, top_n_words=10):
    
    bestTopics = topTopics.sort_values(ascending=False,by=topTopics.columns.values[0]).index.values[:30].tolist() # Get 10 best topics
    topk_least_similar_topics = torch.ones(len(bestTopics),10)
    distx = []
    for k, topic in enumerate(bestTopics):
        topk_least_similar_topics[k] = torch.sort(torch.norm(ell[topic:(topic+1),:].repeat(numTopics,1)-ell,dim=1),descending=True)[1][1:11]
    
        print('Factor {}: Least similar to Factor {}'.format(bestTopics[k], topk_least_similar_topics[k].tolist()))
        tempidx = int(topk_least_similar_topics[k][0])
        #topn_words = torch.sort(gamma[topic,:],descending=True)[1][:top_n_words]
        topn_words = torch.sort(gamma[topic,:],descending=True)[1]
        topn_words_least_topic = torch.sort(gamma[tempidx,:],descending=True)[1][:top_n_words]
        
        print('-' * 89)
        print('Factor {} has these top mutations: '.format(bestTopics[k]))
        vocabx = vocab[topn_words[:20].cpu()]
        print(vocabx)
        tempEGFR = np.where(vocab[topn_words.cpu()]=='ARID1A')
        print('EGFR occurs at location {}'.format(tempEGFR))
        tempKRAS = np.where(vocab[topn_words.cpu()]=='SOX9')
        print('KRAS occurs at location {}'.format(tempKRAS))
        
        distx.append(np.absolute(tempEGFR[0]-tempKRAS[0]))
        print('-' * 89)
        #print('Factor {} has these top mutations: '.format(topk_least_similar_topics[k][0]))
        #print(vocab[topn_words_least_topic.cpu()])
    
    print (np.mean(distx))
        


def display_topics_TML(phi,M,cancer_type,patient_list,tcga_tss_codes,numTopics):
    
    tss_codes = tcga_tss_codes[tcga_tss_codes['ABV'].str.contains(cancer_type)]['TSS'].tolist()
    
    # process codes
    for i in range(len(tss_codes)):
        if len(tss_codes[i]) == 1:
            tss_codes[i] = '-0' + tss_codes[i] + '-'
        else:
            tss_codes[i] = '-' + tss_codes[i] + '-'

    index_of_barcodes = patient_list[patient_list[0].str.contains('|'.join(tss_codes))].index.tolist()
    #tempx = index_of_barcodes[:20]
    phi2 = np.asarray(phi)[index_of_barcodes]
    M_2 = np.asarray(M)[index_of_barcodes]
    #best_topic = np.zeros((numTopics,1)
    best_topic = pd.DataFrame(data=np.zeros((len(index_of_barcodes),numTopics+1)),index=None,columns=[*np.arange(0,60), 'TML'])
    for k in range(len(phi2)):
        phi3 = phi2[k].detach().cpu() # shape num_of_topics x number of mutations in patient k
        temp = torch.sort(phi3,dim=0,descending=True)[1].numpy() # top 5 factors
        best_topic.iloc[k,np.ravel(np.unique(temp[0]))] = 1
        best_topic['TML'].iloc[k] = np.log10(M_2[k])
        
        #print("With TML: {}".format(M_2[k]))
        #print(best_topic)
    
    best_topic2 = pd.melt(best_topic, id_vars=['TML'])
    #plt.yscale('log')
    #plt.tight_layout()
    #plt.figure(figsize=(20,20))
    ax = sns.catplot(x='variable', y='TML', jitter=False, data=best_topic2.query("value != 0"), height=5, aspect=2)
    ax.set(xlabel='Factor Number', ylabel='Log Scale Tumor Mutation Load', title='Mutually Exclusivity of TML and Mutations in COAD')
    ax.savefig('COAD_TMLvsTopics.pdf')
        

    

#plotTopWords(gamma.detach(), vocab)
    
#display_topics(topics, gamma.detach(), ell.detach(), vocab, top_n_words=10, top_n_similar_topics=5)
#dfluad = display_topics_patient_type(phi,'LUAD',mutation_df,tcga_tss_codes,topics)
#norm_topics(dfluad, topics, gamma.detach(), ell.detach().cpu(), vocab)

dfcoad = display_topics_patient_type(phi,'COAD',mutation_df,tcga_tss_codes,topics)
norm_topics(dfcoad, topics, gamma.detach(), ell.detach().cpu(), vocab)

#display_topics_TML(phi,M,'COAD',mutation_df,tcga_tss_codes,topics)

'''
dfcoad = display_topics_patient_type(phi,'COAD',mutation_df,tcga_tss_codes,topics)
dfread = display_topics_patient_type(phi,'READ',mutation_df,tcga_tss_codes,topics)
dfbrca = display_topics_patient_type(phi,'BRCA',mutation_df,tcga_tss_codes,topics)
dfluad = display_topics_patient_type(phi,'LUAD',mutation_df,tcga_tss_codes,topics)
finaldf = pd.concat([dfluad, dfcoad, dfread, dfbrca], axis=1)      
finaldf['index_col'] = finaldf.index
df3 = pd.melt(finaldf,id_vars=['index_col'])
ax = sns.catplot(x='index_col',y='value',hue='variable',kind='bar',data=df3.query("value > 0.1"), height=5,aspect=2)     
ax.set(xlabel='Factor Number', ylabel='Log Scale Count of Factors', title='Number of Factors in a Cancer-Type')
ax.savefig('combined_cancer_box.pdf')
'''    
