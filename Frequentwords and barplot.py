import collections
import nltk
import re
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


nltk.download('punkt')
nltk.download('stopwords')

# Add your custom stopwords to the existing NLTK stopwords list
custom_stopwords = ['litchi','natur','dure','leaf','strawberri','plant','tree','robot', 'robotics', 'agriculture', 'robotic','farthest', 'stroke', 'result', 'moment', 'three', 'joint', 'end',  'consumpt', 'conclus', 'adapt', 'challeng', 'problem', 'appli', 'clearli', 'identifi', 'identifi', 'visibl',  'issu', 'partial', 'full', 'occlus', 'could', 'minim', 'strateg', 'visibl', 'hierarch', 'identif', 'acquir', 'six',  'section',  'wall', 'hough', 'transform', 'cht',  'analysi', 'iter', 'prefer', 'partial', 'initi', 'priorit', 'prove', 'concept', 'taken', 'wellexpos', 'fulli','expos', 'success', 'iter', 'side', 'count', 'addit',  'side', 'remain', 'opposit', 'although', 'simpl', 'intuit', 'uniqu', 'insight', 'achiev', 'show', 'huge', 'potenti',   'compar', 'easier', 'specif', 'access', 'fundament', 'background', 'presenc', 'nontarget', 'adjac',  'v2',  'filter',   'kept', 'remov', 'zfnet', 'vgg16', 'employ',  'highest', 'ap', 'valu', 'indic', 'filter', 'minim', 'applic', 'recent', 'tradit', 'rigid', 'robot', 'attract', 'attent', 'safeti', 'essenti', 'show', 'superior', 'irregular', 'shape', 'conduct', 'adjust', 'variabl', 'effect', 'length', 'vel', 'function', 'mechan', 'tunabl', 'stiff','memori', 'materi', 'requir',   'foldbas', 'finger', 'print', 'experiment', 'model', 'hyperelast', 'properti', 'mathemat', 'finit', 'element', 'model', 'bend', 'behaviour', 'actuat', 'importantli', 'antagonist', 'constraint', 'mechan', 'conform', 'dualmod', 'advanc', 'experiment', 'axi', 'recognit', 'posit', 'owe', 'nois', 'disturb',  'chang',  'shade',  'modifi', 'deep', 'learn',   'cut', 'edg', 'calcul', 'optim', 'clarifi', 'frontlight', 'backlight', 'denois', 'construct', 'basi', 'center', 'symmetri',  'calcul', 'edg', 'equat', 'determin', 'excel', 'balanc',   'percentag', 'satisfi', 'realtim', 'oper', 'evalu',  'techniqu', 'implement', 'select', 'orient',  'requir', 'comput', 'requir', 'knowledg', 'includ',  'measur', 'unit', 'obtain', 'normal', 'contact', 'forc', 'threefing', 'grasp', 'well', 'rotat',  'separ', 'measur', 'calcul', 'vari', 'optimum', 'rel', 'attach', 'varieti', 'semant', 'fundament', 'understand', 'surround', 'enabl', 'acquir', 'rich',  'combin', 'color',  'propos', 'perform', 'fuse',  'critic', 'explor', 'effici', 'textur',  'train', 'sever', 'imbalanc', 'class', 'moreov', 'fusion', 'collect', 'label', 'infer', 'detail', 'experi', 'setup', 'deal', 'noisi',  'miou', '100k200k', 'point', 'real','part', 'distinguish', 'construct',  'map', 'plan',  'motion',  'manipul', 'object',  'background', 'object', 'classifi', 'robust', 'variat', 'among', 'scene', 'classifi', 'primarili', 'soft', 'top',  'bottom',  'hard', 'secondarili', 'five',  'system', 'light', 'develop', 'mitig', 'disturb', 'caus',  'condit', 'success','use',  'nm', 'occur', 'scene', 'includ','construct', 'element', 'support', 'wire', 'remov', 'blue', 'classifi', 'classif', 'regress','cart', 'train', 'pixelbas', 'featur', 'normal', 'differ', 'index', 'strongest', 'select', 'sequenti', 'float', 'forward', 'select',  'new', 'robustandbalanc', 'accuraci', 'perform', 'measur', 'prob', 'introduc',  'featur', 'use', 'render', 'standard', 'deviat', 'reduc', 'compar', 'balanc', 'two', 'approach', 'deriv', 'approach', 'base', 'vs', 'separ', 'class', 'perform', 'slightli', 'better', 'mean', 'detect', 'rate','result', 'insuffici', 'accur', 'suggest', 'improv', 'describ', 'nevertheless', 'first', 'studi', 'report', 'quantit', 'sever', 'vari', 'robot', 'arm', 'avoid', 'collis', 'candid', 'therefor', 'aim', 'local', 'topic', 'far', 'studi', 'control', 'develop', 'capabl', 'wire', 'twist', 'around', 'quantit', 'evalu', 'determin', 'depth',  'lab', 'singl', 'mount','slide', 'record',  'pair', 'small',  'cm', 'consist', 'step', 'includ', 'novel', 'compon', 'adapt', 'threshold', 'cue',  'minimum', 'expect', 'distanc', 'favour', 'moder', 'irradi', 'strong', 'error', 'measur', 'smaller',  'segment', 'detect','increas', 'met', 'requir', 'due', 'movement', 'record', 'probabl', 'use', 'collis', 'robot', 'issu', 'inaccur', 'resolv', 'direct', 'propos', 'futur', 'work', 'regard', 'applic',  'grow', 'along', 'articl', 'data', 'also', 'fashion',  'environ', 'despit', 'avail', 'labor', 'product', 'sens', 'robust', 'agricultur', 'key', 'goal', 'contemporari', 'agricultur', 'dramat', 'increas', 'product', 'sustain', 'face', 'pressur', 'diminish', 'suppli', 'robot', 'acceler', 'advanc', 'datadriven', 'precis', 'farm', 'significantli', 'input', 'provid', 'taskappropri', 'actuat', 'fine',  'resolut', 'highlight', 'distinct', 'challeng', 'impos', 'ground', 'environ', 'character', 'wide', 'variat', 'environment', 'divers', 'complex', 'structur',  'characterist', 'respons', 'exist', 'address', 'present', 'limit', 'possibl', 'discuss', 'observ', 'biolog', 'practic', 'reduc', 'variabl', 'sourc', 'publicli', 'set', 'need', 'percept', 'variabl', 'method', 'howev', 'time', 'use', 'show', 'averag', 'suitabl', 'test',  'total',  'test', 'improv', 'improv', 'optim', 'import', 'collect', 'faster', 'rcnn', 'detect', 'higher', 'speed', 'reliabl', 'step',  'estim', 'often', 'target', 'larg', 'rel', 'spars', 'provid', 'good', 'solut', 'dens', 'distribut', 'propos',  'model', 'adjust', 'predict', 'scale',  'abil', 'dens', 'ensur', 'day',  'bound', 'box', 'label', 'priori', 'box', 'size',  'actual', 'effect', 'f1', 'valu', 'assess', 'ms', 'remit', 'may', 'refer',  'highli',  'signific', 'four', 'pick', 'system',  'pick', 'everi', 'manual', 'destin', 'fresh','human', 'hand', 'extens', 'research', 'past', 'decad', 'mechan', 'commerci', 'concern', 'increas', 'uncertainti', 'avail', 'rise', 'cost', 'unstructur', 'major', 'challeng', 'develop', 'design', 'evalu', 'adopt', 'lowcost', 'assess', 'requir', 'plan', 'manipul', 'function', 'modern', 'planar', 'commerci', 'state', 'workspac', 'modif', 'criteria', 'thoroughli', 'defin', 'report', 'help', 'guid', 'enhanc', 'per', 'seven', 'degre', 'freedom', 'attempt', 'overal', 'success', 'includ', 'integr', 'addit', 'method', 'differ', 'process', 'process', 'achiev', 'provid', 'branch', 'obtain', 'inform', 'precis', 'thu', 'point', 'cluster', 'segment', 'posit', 'bunchtyp', 'strategi', 'roughli', 'bunch', 'rout', 'locat', 'precis', 'closer', 'latter', 'amount', 'process', 'detail', 'improv', 'longclos', 'coordin', 'control', 'intel', 'combin', 'cloud', 'object', 'bunch', 'long', 'deduc', 'sequenc', 'reach', 'closedist', 'mask', 'instanc', 'view', 'mask', 'dual', 'line', 'guid', 'studi', 'took', 'account', 'advantag', 'disadvantag', 'experi', 'complet', 'abl', 'locat', 'theoret', 'technic', 'design', 'angl', 'architectur', 'architectur', 'task', 'decreas', 'cost', 'optim', 'specif', 'task', 'work', 'sinc', 'defin', 'optimis', 'object', 'present', 'methodolog', 'simultan', 'kinemat', 'demonstr', 'exampl', 'structur', 'found', 'number', 'shape', 'train', 'central', 'leader', 'tall', 'spindl', 'indic', 'minimis', 'prefer', 'row', 'influenc', 'platform', 'chosen', 'analys', 'fast', 'platform', 'advantag', 'posit', 'near', 'slow', 'addit', 'tilt', 'simul', 'model', 'creat', 'lsystem', 'simul', 'nearli', 'allow', 'design', 'choos', 'combin', 'grasp', 'forc', 'experi', 'limit', 'present', 'one', 'respect', 'comprehens', 'achiev', 'shortag', 'sever', 'rapid', 'current', 'ineffici', 'detach', 'achiev', 'cup', 'customis', 'interfer', 'complianc', 'exert', 'evalu', 'adhes', 'activ', 'passiv', 'mode', 'simultan', 'final', 'implement', 'detach', 'compact', 'compliant', 'gener', 'valid', 'damag', 'grow', 'power', 'complex', 'set', 'oper', 'establish', 'rang', 'problem', 'solv', 'second', 'analyz', 'low', 'part', 'narrow', 'ridg', 'great', 'aim', 'space', 'ridg', 'prrprr', 'realiz', 'automat', 'establish', 'condit', 'oper', 'order', 'joint', 'singular', 'within', 'index', 'greater', 'accord', 'simul', 'cycl', 'less', 'wa','algorithm','thi','ha','smooth', 'uniform', 'peak', 'veloc', 'lower', 'take', 'threejoint', 'need', 'overcom', 'reach']

# Rest of your custom stopwords...

def get_word_frequency(document):
    document = document.lower()
    document = re.sub(r'[^\w\s]', '', document)
    document = re.sub(r'\b\d+\b|\b\w{1}\b', '', document)

    words = word_tokenize(document)
    
    # Apply stemming using NLTK's Porter stemmer
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    # Combine NLTK stopwords with custom stopwords
    all_stopwords = set(stopwords.words('english') + custom_stopwords)
    
    words = [word for word in words if word not in all_stopwords]
    word_freq = collections.Counter(words)
    return word_freq

def create_frequency_table(documents):
    term_document_freq = {}

    for doc_idx, doc in enumerate(documents, start=1):
        word_frequency = get_word_frequency(doc)
        
        for term, freq in word_frequency.items():
            if term not in term_document_freq:
                term_document_freq[term] = {}
            term_document_freq[term][f"Document {doc_idx}"] = freq
    
    return term_document_freq

# Example list of documents
f=pd.read_excel(r"D:\STUDY\Sem2\MINI_project\abstract_mainpath.xlsx")
documents = (f["Abstract"])

# Create frequency table
frequency_table = create_frequency_table(documents)

# Get the document order
document_order = [f"Document {doc_idx}" for doc_idx in range(1, len(documents) + 1)]

# Convert the frequency table to a pandas DataFrame with specified column order
df = pd.DataFrame.from_dict(frequency_table, orient='index', columns=document_order)

# Fill empty cells with zeros
df = df.fillna(0)

# Sum the frequency across all documents
df['Total'] = df.sum(axis=1)

# Select the most frequent words
num_most_common = 10  # Choose the number of most frequent words to consider
most_common_words = df['Total'].nlargest(num_most_common).index

# Filter the DataFrame to only include the most common words
df_common_words = df.loc[most_common_words]

# Create a bar plot
plt.figure(figsize=(10, 6))
df_common_words.drop(columns='Total').plot(kind='bar')
plt.title(f"Top {num_most_common} Most Frequent Words Across Articles")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.legend(title="Documents", loc='upper left')  # Specify the location of the legend
plt.tight_layout()

plt.show()
