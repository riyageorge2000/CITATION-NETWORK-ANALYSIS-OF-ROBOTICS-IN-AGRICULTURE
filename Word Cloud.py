import nltk
import re
import pandas as pd
import numpy as np
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('punkt')  # Download NLTK data (if not done already)
nltk.download('stopwords')  # Download stopwords data (if not done already)

def count_word_frequency(documents, custom_stopwords=[]):
    word_frequency = {}
    stemmer = PorterStemmer()
    
    # Combine NLTK's stopwords with custom stopwords
    all_stopwords = set(stopwords.words('english')) | set(custom_stopwords)
    
    for document in documents:
        if isinstance(document, str):  # Check if the document is a string
            words = word_tokenize(document)
            for word in words:
                word = word.lower()
                
                # Remove symbols, characters, and numbers
                word = re.sub(r'[^a-zA-Z]', '', word)
                
                if word:  # Check if the word is not empty
                    stemmed_word = stemmer.stem(word)
                    if stemmed_word not in all_stopwords:  # Check if the stemmed word is not a stopword
                        if stemmed_word in word_frequency:
                            word_frequency[stemmed_word] += 1
                        else:
                            word_frequency[stemmed_word] = 1
    
    return word_frequency

# Load Excel file
file_path = r"D:\STUDY\Sem2\MINI_project\wordcloud 1.xlsx"
f = pd.read_excel(file_path)

# Get the abstracts column
documents = f["Abstract"]

# Custom stopwords to be added
custom_stopwords =  [
    'robotic', 'agricultural', 'robot', 'time', 'based', 'derived', 'additional', 'example', 'discussed', 'used',
    'belongs', 'consists', 'determine', 'important', 'problem', 'improved', 'best', 'since', 'use', 'perform',
    'active', 'new', 'two', 'effect', 'state', 'efficient', 'manipulate', 'locate', 'different', 'given', 'also',
    'one', 'apply', 'present','paper','method','develop', 'describe', 'apply', 'environment', 'determine', 'find', 'consider', 'known',
    'consist', 'minimum', 'require', 'provide', 'show', 'obtain', 'may', 'change', 'position', 'initial', 'alternative',
    'found', 'discuss', 'associate', 'often', 'process', 'along', 'define', 'fastest', 'addition', 'chapter',
    'like', 'common', 'exist', 'later', 'apnea', 'onset', 'main', 'affect', 'indeed', 'welfare', 'example', 'belong',
    'goal', 'exceed', 'previous','combin','process','show', 'appear', 'major', 'important', 'contain', 'well', 'point', 'solve', 'achieve',
    'extension', 'reveal', 'actuate', 'practice', 'depend', 'address', 'due', 'take', 'final', 'event', 'good',
    'illustrate', 'comparison', 'reason', 'refer','achiev','result','respect', 'upon', 'select', 'value', 'move', 'place', 'assign', 'take',
    'operate', 'lrsw', 'four', 'via', 'various','estim', 'character', 'communicate', 'interest', 'prenatal', 'function',
    'express', 'suggest', 'review', 'recent', 'basic', 'update', 'hand', 'already', 'part', 'direct', 'current',
    'although', 'joint', 'however', 'three', 'case', 'concern', 'manage', 'start', 'v', 'wide', 'focus', 'whose',
    'give', 'expect', 'play', 'increasingly', 'daily', 'live', 'combine', 'organize', 'create', 'rapidly', 'emerge',
    'beside', 'edit', 'image', 'taskplan', 'n', 'every', 'rather', 'many', 'closelyrel', 'would', 'half', 'highly',
    'examine', 'go', 'approach', 'equal', 'introduce', 'latter', 'correspond', 'either', 'made', 'denote', 'begin',
    'maximum', 'observe', 'thus', 'unlike', 'able', 'exhaust', 'across', 'bed', 'reach', 'adjacent', 'assume', 'get',
    'way', 'condition', 'intervene', 'greedy', 'k', 'shown', 'handle', 'enable', 'put', 'forth', 'hold', 'map',
    'allow', 'include', 'participate', 'age', 'utilize', '15', 'complete', 'nine', 'discrete', 'nare', 'feature',
    'extract', 'precede', 'agree', 'previous', 'environment', 'risk', 'disturb', 'nonselect', 'induce', 'restore',
    'anxiety', 'impact', 'higher', '6', 'il6', 'might', 'subsequent', 'underlie', 'expenditure', 'possible',
    'compete', 'consider', 'done', 'subject', 'turn', 'purpose', 'earlier', 'six', 'response', 'competitive',
    'involve', 'empirical', 'valid', 'conclude', 'future', 'simultaneously', 'status', 'significantly', 'last',
    'decade', 'among', 'definite', 'represent', 'share', 'large', 'kind', 'suffer', 'hence', 'small', 'context',
    'prefer', 'board', 'column', 'constrain', 'enhance', 'shortest', 'drive', 'available', 'even', 'avoid',
    'necessarily', 'another', 'first', 'stanford', 'demonstrate', 'usual', 'less', 'straightline', 'uncertain',
    'scenario', '49', 'continue', 'mode', 'therefore', 'attain', '100', 'varied', 'forward', 'thereby', 'reality',
    'make', 'valuable', 'realize', 'need', 'meaning', 'stage', 'relate', 'creation', 'year', 'raise', 'overuse',
    'adopt', 'strongly', 'attitude', 'poorly', 'aim', 'understand', 'perception', 'questionnaire', 'administer',
    '86', 'conduct', 'region', 'distinct', 'aware', 'aspect', 'negative', 'satisfaction', 'coverage', 'source',
    'information', 'confidence', 'follow', 'amount', 'monitor', 'lack', 'full', 'influence', 'efficacy', 'robot',
    'robotics', 'agriculture', 'farthest', 'stroke', 'result', 'moment', 'end', 'consumption', 'conclusion',
    'adapt', 'challenge', 'problem', 'application', 'clearly', 'identify', 'visible', 'issue', 'partial', 'occlusion',
    'could', 'minimum', 'strategy', 'hierarchy', 'acquire', 'six', 'section', 'wall', 'hough', 'transform', 'cht',
    'analysis', 'iteration', 'initial', 'prioritize', 'prove', 'concept', 'well', 'expose', 'side', 'addition',
    'remain', 'opposite', 'simple', 'intuitive', 'unique', 'insight', 'achieve', 'show', 'huge', 'potential',
    'comparison', 'easier', 'specific', 'access', 'fundamental', 'background', 'presence', 'nontarget', 'adjacent',
    'v2', 'filter', 'kept', 'remove', 'zfnet', 'vgg16', 'employ', 'highest', 'ap', 'value', 'indicative', 'minimal',
    'application', 'recent', 'traditional', 'rigid', 'attract', 'attention', 'safety', 'essential', 'superior',
    'irregular', 'shape', 'conduct', 'adjust', 'variable', 'effect', 'length', 'velocity', 'function', 'mechanism',
    'tunable', 'stiff', 'memory', 'material', 'require', 'foldbas', 'finger', 'print', 'experiment', 'model',
    'hyperelastic', 'property', 'mathematical', 'finite', 'element', 'bend', 'behavior', 'actuate', 'important',
    'antagonist', 'constraint', 'conform', 'dualmod', 'advance', 'axi', 'recognition', 'positive', 'noise', 'disturb',
    'change', 'shade', 'modify', 'deep', 'cut', 'edge', 'calculate', 'optimize', 'clarify', 'frontlight',
    'backlight', 'denoise', 'construct', 'base', 'center', 'symmetry', 'equation', 'determine', 'excel', 'balance',
    'percentage', 'satisfy', 'realtime', 'operate', 'evaluate', 'technique', 'implement', 'select', 'orient',
    'require', 'compute', 'measurement', 'unit', 'obtain', 'normal', 'contact', 'force', 'threefing', 'grasp',
    'rotate', 'separate', 'measurement', 'calculate', 'variety', 'optimum', 'related', 'attach', 'semantic',
    'fundamental', 'understand', 'surround', 'enable', 'acquire', 'rich', 'combine', 'color', 'propose', 'perform',
    'fuse', 'critical', 'explore', 'efficient', 'texture', 'train', 'several', 'imbalance', 'class', 'moreover',
    'fusion', 'collect', 'label', 'infer', 'detail', 'experience', 'setup', 'deal', 'noisy', 'miou', '100k200k',
    'point', 'real', 'part', 'distinguish', 'construct', 'map', 'plan', 'motion', 'manipulate', 'object', 'background',
    'classifi', 'robust', 'variation', 'among', 'scene', 'primarily', 'soft', 'top', 'bottom', 'hard', 'secondarily',
    'five', 'system', 'light', 'develop', 'mitigate', 'disturb', 'cause', 'condition', 'success', 'nm', 'occur',
    'scene', 'include', 'construct', 'element', 'support', 'wire', 'remove', 'blue', 'pixelbas', 'feature', 'normal',
    'different', 'index', 'strongest', 'select', 'sequential', 'float', 'forward', 'select', 'new', 'robustandbalanc',
    'accuracy', 'perform', 'measure', 'prob', 'introduce', 'feature', 'use', 'render', 'standard', 'deviation',
    'reduce', 'compare', 'balance', 'two', 'approach', 'derive', 'approach', 'base', 'vs', 'separate', 'class',
    'perform', 'slightly', 'better', 'mean', 'detect', 'rate', 'result', 'insufficient', 'accurate', 'suggest',
    'improve', 'describe', 'nevertheless', 'first', 'study', 'report', 'quantitative', 'several', 'variation', 'robot',
    'arm', 'avoid', 'collision', 'candidate', 'therefore', 'aim', 'local', 'topic', 'far', 'study', 'control',
    'develop', 'capability', 'wire', 'twist', 'around', 'quantitative', 'evaluate', 'determine', 'depth', 'lab',
    'single', 'mount', 'slide', 'record', 'pair', 'small', 'cm', 'consist', 'step', 'include', 'novel', 'component',
    'adapt', 'threshold', 'cue', 'minimum', 'expect', 'distance', 'favour', 'moderate', 'irradiation', 'strong',
    'error', 'measure', 'small', 'segment', 'detect', 'increase', 'meet', 'require', 'due', 'movement', 'record',
    'probably', 'use', 'collision', 'robot', 'issue', 'inaccurate', 'resolve', 'direct', 'propose', 'future', 'work',
    'regard', 'application', 'grow', 'along', 'article', 'data', 'also', 'fashion', 'despite', 'availability',
    'labor', 'product', 'sense', 'robust', 'agriculture', 'key', 'goal', 'contemporary', 'agriculture', 'dramatic',
    'increase', 'product', 'sustain', 'face', 'pressure', 'diminish', 'supply', 'robot', 'accelerate', 'advance',
    'datadriven', 'precise', 'farm', 'significantly', 'input', 'provide', 'taskappropriate', 'actuate', 'fine',
    'resolution', 'highlight', 'distinct', 'challenge', 'impose', 'ground', 'environment', 'character', 'wide',
    'variation', 'environment', 'diverse', 'complex', 'structure', 'characteristic', 'response', 'exist', 'address',
    'present', 'limit', 'possible', 'discuss', 'observe', 'biological', 'practice', 'reduce', 'variable', 'source',
    'publicly', 'set', 'need', 'percept', 'variable', 'method', 'however', 'time', 'use', 'show', 'average',
    'suitable', 'test', 'total', 'test', 'improve', 'optimize', 'important', 'collect', 'faster',  'detect',
    'higher', 'speed', 'reliable', 'step', 'estimate', 'often', 'target', 'large', 'relative', 'sparse', 'provide',
    'good', 'solution', 'dense', 'distribution', 'propose', 'model', 'adjust', 'predict', 'scale', 'ability',
    'dense', 'ensure', 'day', 'bound', 'box', 'label', 'priori', 'box', 'size', 'actual', 'effect', 'f1', 'value',
    'assess', 'ms', 'remit', 'may', 'refer', 'highly', 'significant', 'four', 'pick', 'system', 'everi', 'manual',
    'destination', 'fresh', 'human', 'hand', 'extend', 'research', 'past', 'decade', 'mechanism', 'commercial',
    'concern', 'increase', 'uncertainty', 'availability', 'rise', 'cost', 'unstructured', 'major', 'challenge',
    'develop', 'design', 'evaluate', 'adopt', 'lowcost', 'assess', 'require', 'plan', 'manipulate', 'function',
    'modern', 'planar', 'commercial', 'state', 'workspace', 'modify', 'criterion', 'thoroughly', 'define', 'report',
    'help', 'guide', 'enhance', 'per', 'seven', 'degree', 'freedom', 'attempt', 'overall', 'success', 'include',
    'integrate', 'additional', 'different', 'process', 'achieve', 'provide', 'branch', 'obtain', 'information',
    'precise', 'thus', 'point', 'cluster', 'segment', 'positive', 'bunchtyp', 'strategy', 'roughly', 'bunch',
    'route', 'locate', 'precise', 'closer', 'latter', 'amount', 'process', 'detail', 'improve', 'longclos', 'coordinate',
    'control', 'intel', 'd435i', 'combine', 'cloud', 'object', 'bunch', 'long', 'deduce', 'sequence', 'reach',
    'closedist', 'mask', 'instance', 'view', 'dual', 'line', 'guide', 'study', 'take', 'account', 'advantage',
    'disadvantage', 'experience', 'complete', 'able', 'locate', 'theoretical', 'technique', 'design', 'angle',
    'architecture', 'task', 'decrease', 'cost', 'optimize', 'specific', 'task', 'work', 'since', 'define', 'optimize',
    'object', 'methodology', 'simultaneous', 'kinematic', 'demonstrate', 'example', 'structure', 'find', 'number',
    'shape', 'train', 'central', 'leader', 'tall', 'spindle', 'indicative', 'minimize', 'prefer', 'row', 'influence',
    'platform', 'choose', 'analyze', 'fast', 'platform', 'advantage', 'positive', 'near', 'slow', 'additional',
    'tilt', 'simulate', 'model', 'create', 'lsystem', 'robot','simulate', 'nearly', 'allow', 'design', 'choose', 'combine',
    'grasp', 'apply', 'force', 'experience', 'limit', 'present', 'one', 'respect', 'comprehensive', 'achieve',
    'shortage', 'several', 'rapid', 'current', 'inefficient', 'detach', 'achieve', 'cup', 'customize', 'interfere',
    'compliance', 'exert', 'evaluate', 'adhesion', 'active', 'passive', 'mode', 'simultaneous', 'final', 'implement',
    'detach', 'compact', 'compliant', 'generate', 'validate', 'damage', 'grow', 'power', 'complex', 'set', 'operate',
    'establish', 'range', 'problem', 'solve', 'second', 'analyze', 'low', 'part', 'narrow', 'ridge', 'great', 'aim',
    'space', 'ridg', 'prrprr', 'realize', 'automate', 'establish', 'condition', 'operate', 'order', 'joint', 'singular',
    'within', 'index', 'greater', 'accord', 'simulate', 'cycle', 'less', 'wa', 'algorithm', 'smooth', 'uniform',
    'peak', 'velocity', 'lower', 'take', 'threejoint', 'need', 'overcome', 'reach','robotic','agricultural','robot','time',
    'based','derived','additional','example','discussed','used','belongs','consists','determine','important','problem',
    'improved','best','since','use', 'agricultur',  'perform',  'activ', 'new', 'two', 'effect', 'state', 'effici', 'manipul',
    'locat', 'differ',  'given', 'also', 'one',  'applic', 'present','describ',  'appli', 'environ',  'determin', 'find',
    'consid', 'known', 'consist', 'minim', 'requir',  'provid', 'show',  'obtain', 'may', 'chang', 'posit',  'initi', 
    'altern', 'found', 'discuss', 'associ',  'st',  'often', 'process', 'along', 'defin',  'fastest',  'addit', 'chapter', 
    'aris', 'like', 'common', 'exist','later',  'apnea', 'onset', 'main','affect','inde', 'welfar',  'exampl', 'belong',
    'goal',  'exceed', 'previous', 'appear', 'major', 'import', 'contain', 'well','point', 'solv', 'achiev','extens', 
    'reveal', 'actuat','practic', 'depend', 'address', 'due', 'taken', 'final','event', 'good', 'illustr', 'comparison',
    'reason', 'refer', 'upon',  'select', 'valu', 'move', 'place',  'assign', 'take',  'oper', 'lrsw',  'four', 'via',
    'variou',  'character', 'commun', 'interest', 'prenat', 'function', 'express', 'suggest', 'review', 'recent', '1977',
    'basic', 'updat', 'hand', 'alreadi',  'part', 'direct', 'current', 'assess', 'although', 'joint', 'howev',  'three',
    'case', 'concern', 'manag',  'g', 'start', 'v',  'wide', 'focu',  'whose', 'give', 'expect','play', 'increasingli',
    'daili', 'live', 'combin', 'organiz', 'creat',  'rapidli','emerg', 'besid',  'edit',  'imag', '85', 'taskplan', 'n', 
    'pass','everi', 'rather',  'mani', 'closelyrel', 'would', '25', 'half','highli', 'examin', 'go', 'approach','equal',
    'introduc', 'latter', 'correspond', 'either', 'made', 'denot',  'begin', 'maxim', 'observ', 'thu', 'unlik',  'abl',
    'exhaust',  'across', 'bed', 'reach', 'adjac', 'assum', 'get', 'way', 'condit', 'interv', 'greedi', 'k', 'shown', 
    'handl', 'enabl',  'put', 'forth', 'hold', 'map',  'allow', 'includ',  'particip', 'age',  'util',  '15','complet', 
    'nine', 'discret','nare',  'featur', 'extract', 'preced', 'agre', 'previou','environment',  'risk', 'disturb',
    'nonselect', 'induc', 'restor',  'anxieti',  'impact', 'higher', '6', 'il6','might',  'subsequ', 'underli',
    'expenditur', 'possibl', 'compet', 'consider', 'done', 'subject',  'turn', 'purpos','earlier','six',  'respons', 
    'competit', 'involv', 'empir', 'valid', 'conclud', 'futur', 'simultan', 'statu', 'significantli', 'last', 'decad', 
    'among',  'definit', 'repres', 'share', 'larg', 'kind', 'suffer','henc',  'small', 'context',  'prefer', 'board',
    'column', 'constrain', 'enhanc','shortest', 'drive', 'avail', 'even', 'avoid', 'necessarili',  'anoth', 'first',
    'pac', 'stanford', 'demonstr', 'usual', 'less', 'straightlin','uncertain','scenario', '49', 'continu', 'mode', 
    'therefor', 'attain', '100', 'vari', 'forward', 'therebi', 'realiti', 'make', 'valuabl', 'realiz','need', 'meaning', 
    'stage', 'relat', 'creation', 'year', 'rais', 'overus', 'adopt', 'strongli', 'attitud', 'poorli', 'aim', 'understand',
    'percept', 'questionnair', 'administ', '86',  'conduct', 'region', 'distinct',   'awar', 'aspect', 'neg', 'satisfact', 
    'coverag', 'sourc', 'inform', 'confid', 'follow', 'amount', 'monitor', 'lack', 'full', 'influenc', 'efficaci','robot', 
    'robotics', 'agriculture', 'robotic','farthest', 'stroke', 'result', 'moment', 'three', 'joint', 'end',  'consumpt',
    'conclus', 'adapt', 'challeng', 'problem', 'appli', 'clearli', 'identifi', 'identifi', 'visibl',  'issu', 'partial', 
    'full', 'occlus', 'could', 'minim', 'strateg', 'visibl', 'hierarch', 'identif', 'acquir', 'six',  'section',  'wall', 
    'hough', 'transform', 'cht',  'analysi', 'iter', 'prefer', 'partial', 'initi', 'priorit', 'prove', 'concept', 'taken', 
    'wellexpos', 'fulli','expos', 'success', 'iter', 'side', 'count', 'addit',  'side', 'remain', 'opposit', 'although', 
    'simpl', 'intuit', 'uniqu', 'insight', 'achiev', 'show', 'huge', 'potenti',   'compar', 'easier', 'specif', 'access',
    'fundament', 'background', 'presenc', 'nontarget', 'adjac',  'v2',  'filter',   'kept', 'remov', 'zfnet', 'vgg16',
    'employ',  'highest', 'ap', 'valu', 'indic', 'filter', 'minim', 'applic', 'recent', 'tradit', 'rigid', 'robot',
    'attract', 'attent', 'safeti', 'essenti', 'show', 'superior', 'irregular', 'shape', 'conduct', 'adjust', 'variabl', 
    'effect', 'length', 'vel', 'function', 'mechan', 'tunabl', 'stiff','memori', 'materi', 'requir',   'foldbas', 'finger', 
    'print', 'experiment', 'model', 'hyperelast', 'properti', 'mathemat', 'finit', 'element', 'model', 'bend', 'behaviour', 
    'actuat', 'importantli', 'antagonist', 'constraint', 'mechan', 'conform', 'dualmod', 'advanc', 'experiment', 'axi', 
    'recognit', 'posit', 'owe', 'nois', 'disturb',  'chang',  'shade',  'modifi', 'deep',   'cut', 'edg', 'calcul', 
    'optim', 'clarifi', 'frontlight', 'backlight', 'denois', 'construct', 'basi', 'center', 'symmetri',  'calcul', 'edg', 
    'equat', 'determin', 'excel', 'balanc',   'percentag', 'satisfi', 'realtim', 'oper', 'evalu',  'techniqu', 'implement',
    'select', 'orient',  'requir', 'comput', 'requir', 'knowledg', 'includ',  'measur', 'unit', 'obtain', 'normal', 'contact', 
    'forc', 'threefing', 'grasp', 'well', 'rotat',  'separ', 'measur', 'calcul', 'vari', 'optimum', 'rel', 'attach', 'varieti',
    'semant', 'fundament', 'understand', 'surround', 'enabl', 'acquir', 'rich',  'combin', 'color',  'propos', 'perform',
    'fuse',  'critic', 'explor', 'effici', 'textur',  'train', 'sever', 'imbalanc', 'class', 'moreov', 'fusion', 'collect',
    'label', 'infer', 'detail', 'experi', 'setup', 'deal', 'noisi',  'miou', '100k200k', 'point', 'real','part', 'distinguish',
    'construct',  'map', 'plan',  'motion',  'manipul', 'object',  'background', 'object', 'classifi', 'robust', 'variat', 
    'among', 'scene', 'classifi', 'primarili', 'soft', 'top',  'bottom',  'hard', 'secondarili', 'five',  'system',
    'light', 'develop', 'mitig', 'disturb', 'caus',  'condit', 'success','use',  'nm', 'occur', 'scene', 'includ',
    'construct', 'element', 'support', 'wire', 'remov', 'blue', 'classifi', 'classif', 'regress','cart', 'train', 
    'pixelbas', 'featur', 'normal', 'differ', 'index', 'strongest', 'select', 'sequenti', 'float', 'forward',
    'select',  'new', 'robustandbalanc', 'accuraci', 'perform', 'measur', 'prob', 'introduc',  'featur', 'use', 
    'render', 'standard', 'deviat', 'reduc', 'compar', 'balanc', 'two', 'approach', 'deriv', 'approach', 'base', 'vs',
    'separ', 'class', 'perform', 'slightli', 'better', 'mean', 'detect', 'rate','result', 'insuffici', 'accur', 'suggest', 
    'improv', 'describ', 'nevertheless', 'first', 'studi', 'report', 'quantit', 'sever', 'vari', 'robot', 'arm', 'avoid', 
    'collis', 'candid', 'therefor', 'aim', 'local', 'topic', 'far', 'studi', 'control', 'develop', 'capabl', 'wire', 'twist', 
    'around', 'quantit', 'evalu', 'determin', 'depth',  'lab', 'singl', 'mount','slide', 'record',  'pair', 'small',  'cm', 
    'consist', 'step', 'includ', 'novel', 'compon', 'adapt', 'threshold', 'cue',  'minimum', 'expect', 'distanc', 'favour', 
    'moder', 'irradi', 'strong', 'error', 'measur', 'smaller',  'segment', 'detect','increas', 'met', 'requir', 'due', 
    'movement', 'record', 'probabl', 'use', 'collis', 'robot', 'issu', 'inaccur', 'resolv', 'direct', 'propos', 'futur', 
    'work', 'regard', 'applic',  'grow', 'along', 'articl', 'data', 'also', 'fashion',  'environ', 'despit', 'avail', 
    'labor', 'product', 'sens', 'come','robust', 'agricultur', 'key', 'goal', 'contemporari', 'agricultur', 'dramat', 'increas', 'product', 'sustain', 'face', 'pressur', 'diminish', 'suppli', 'robot', 'acceler', 'advanc', 'datadriven', 'precis', 'farm', 'significantli', 'input', 'provid', 'taskappropri', 'actuat', 'fine',  'resolut', 'highlight', 'distinct', 'challeng', 'impos', 'ground', 'environ', 'character', 'wide', 'variat', 'environment', 'divers', 'complex', 'structur',  'characterist', 'respons', 'exist', 'address', 'present', 'limit', 'possibl', 'discuss', 'observ', 'biolog', 'practic', 'reduc', 'variabl', 'sourc', 'publicli', 'set', 'need', 'percept', 'variabl', 'method', 'howev', 'time', 'use', 'show', 'averag', 'suitabl', 'test',  'total',  'test', 'improv', 'improv', 'optim', 'import', 'collect', 'faster',  'detect', 'higher', 'speed', 'reliabl', 'step',  'estim', 'often', 'target', 'larg', 'rel', 'spars', 'provid', 'good', 'solut', 'dens', 'distribut', 'propos',  'model', 'adjust', 'predict', 'scale',  'abil', 'dens', 'ensur', 'day',  'bound', 'box', 'label', 'priori', 'box', 'size',  'actual', 'effect', 'f1', 'valu', 'assess', 'ms', 'remit', 'may', 'refer',  'highli',  'signific', 'four', 'pick', 'system',  'pick', 'everi', 'manual', 'destin', 'fresh','human', 'hand', 'extens', 'research', 'past', 'decad', 'mechan', 'commerci', 'concern', 'increas', 'uncertainti', 'avail', 'rise', 'cost', 'unstructur', 'major', 'challeng', 'develop', 'design', 'evalu', 'adopt', 'lowcost', 'assess', 'requir', 'plan', 'manipul', 'function', 'modern', 'planar', 'commerci', 'state', 'workspac', 'modif', 'criteria', 'thoroughli', 'defin', 'report', 'help', 'guid', 'enhanc', 'per', 'seven', 'degre', 'freedom', 'attempt', 'overal', 'success', 'includ', 'integr', 'addit', 'method', 'differ', 'process', 'process', 'achiev', 'provid', 'branch', 'obtain', 'inform', 'precis', 'thu', 'point', 'cluster', 'segment', 'posit', 'bunchtyp', 'strategi', 'roughli', 'bunch', 'rout', 'locat', 'precis', 'closer', 'latter', 'amount', 'process', 'detail', 'improv', 'longclos', 'coordin', 'control', 'intel', 'd435i', 'combin', 'cloud', 'object', 'bunch', 'long', 'deduc', 'sequenc', 'reach', 'closedist', 'mask', 'instanc', 'view', 'mask', 'dual', 'line', 'guid', 'studi', 'took', 'account', 'advantag', 'disadvantag', 'experi', 'complet', 'abl', 'locat', 'theoret', 'technic', 'design', 'angl', 'architectur', 'architectur', 'task', 'decreas', 'cost', 'optim', 'specif', 'task', 'work', 'sinc', 'defin', 'optimis', 'object', 'present', 'methodolog', 'simultan', 'kinemat', 'demonstr', 'exampl', 'structur', 'found', 'number', 'shape', 'train', 'central', 'leader', 'tall', 'spindl', 'indic', 'minimis', 'prefer', 'row', 'influenc', 'platform', 'chosen', 'analys', 'fast', 'platform', 'advantag', 'posit', 'near', 'slow', 'addit', 'tilt', 'simul', 'model', 'creat', 'lsystem', 'simul', 'nearli', 'allow', 'design', 'choos', 'combin', 'grasp', 'appl', 'forc', 'experi', 'limit', 'present', 'one', 'respect', 'comprehens', 'achiev', 'shortag', 'sever', 'rapid', 'current', 'ineffici', 'detach', 'achiev', 'cup', 'customis', 'interfer', 'complianc', 'exert', 'evalu', 'adhes', 'activ', 'passiv', 'mode', 'simultan', 'final', 'implement', 'detach', 'compact', 'compliant', 'gener', 'valid', 'damag', 'grow', 'power', 'complex', 'set', 'oper', 'establish', 'rang', 'problem', 'solv', 'second', 'analyz', 'low', 'part', 'narrow', 'ridg', 'great', 'aim', 'space', 'ridg', 'prrprr', 'realiz', 'automat', 'establish', 'condit', 'oper', 'order', 'joint', 'singular', 'within', 'index', 'greater', 'accord', 'simul', 'cycl', 'less', 'wa','algorithm','thi','ha','smooth', 'uniform', 'peak', 'veloc', 'lower', 'take', 'threejoint', 'need', 'overcom', 'reach']



# Count word frequency
word_frequency = count_word_frequency(documents, custom_stopwords)

# Print word frequency
for word, freq in word_frequency.items():
    print(f"{word}: {freq}")

# Create a WordCloud
mask_image = np.array(Image.open(r"D:\STUDY\Sem2\MINI-PROJECT\circle.png")) 
wordcloud = WordCloud(width=800, height=800, background_color='white', max_words=60, mask=mask_image).generate_from_frequencies(word_frequency)

# Plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
