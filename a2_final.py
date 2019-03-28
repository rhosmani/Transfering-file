# coding: utf-8

"""
CS579: Assignment 2

In this assignment, you will build a text classifier to determine whether a
movie review is expressing positive or negative sentiment. The data come from
the website IMDB.com.

You'll write code to preprocess the data in different ways (creating different
features), then compare the cross-validation accuracy of each approach. Then,
you'll compute accuracy on a test set and do some analysis of the errors.

The main method takes about 40 seconds for me to run on my laptop. Places to
check for inefficiency include the vectorize function and the
eval_all_combinations function.

Complete the 14 methods below, indicated by TODO.

As usual, completing one method at a time, and debugging with doctests, should
help.
"""

# No imports allowed besides these.
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request


def download_data():
    """ Download and unzip data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/8oehplrobcgi9cq/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
    """
    Walks all subdirectories of this path and reads all
    the text files and labels.
    DONE ALREADY.

    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.

    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.

    >>> tokenize(" Hi there! Isn't this fun?", keep_internal_punct=False)
    array(['hi', 'there', 'isn', 't', 'this', 'fun'], dtype='<U5')
    >>> tokenize("Hi there! Isn't this fun? ", keep_internal_punct=True)
    array(['hi', 'there', "isn't", 'this', 'fun'], dtype='<U5')
    """
    ###TODO
    tokens =[]
    if keep_internal_punct == True:
        split_doc = doc.lower().split()
        for word in split_doc:
            if not all(character in string.punctuation for character in word):
                word = word.strip(string.punctuation)
                tokens.append(word)
        tokenized = np.array(tokens)
    elif keep_internal_punct == False:
        tokens = re.sub('\W+', ' ', doc.lower()).split()
        tokenized = np.array(tokens)
    return tokenized
    pass


stop_words = set([ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ])

def tokenize_and_remove_stopwords(doc, keep_internal_punct=False):

    tokens =[]
    if keep_internal_punct == True:
        split_doc = doc.lower().split()
        for word in split_doc:
            if not all(character in string.punctuation for character in word):
                word = word.strip(string.punctuation)
                if word not in stop_words:
                	tokens.append(word)
        tokenized = np.array(tokens)
    elif keep_internal_punct == False:
        tokens = re.sub('\W+', ' ', doc.lower()).split()
        tokens1 = []
        for word in tokens:
        	if word not in stop_words:
        		tokens1.append(word)
        tokenized = np.array(tokens1)
    return tokenized


def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_features(['hi', 'there', 'hi'], feats)
    >>> sorted(feats.items())
    [('token=hi', 2), ('token=there', 1)]
    """
    ###TODO
    counterlist = Counter(["token=" + token for token in tokens])
    for word, count in counterlist.items():
    	feats[word] = count
    pass


def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.

    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)

    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]
    """
    ###TODO
    windows = []
    combinationList = []
    for i in range(len(tokens) - k + 1):
    	windows.append(tokens[i : i + k])
    for window in windows:
    	for combo in list(combinations(window, 2)):
    		combinationList.append(combo[0] + '__' + combo[1])
    counterlist = Counter(['token_pair=' + token_pair for token_pair in combinationList])
    for word, count in counterlist.items():
    	feats[word] = count
    pass


neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])
# neg_words = set(['2-faced','2-faces','abnormal','abolish','abominable','abominably','abominate','abomination','abort','aborted','aborts','abrade','abrasive','abrupt','abruptly','abscond','absence','absent-minded','absentee','absurd','absurdity','absurdly','absurdness','abuse','abused','abuses','abusive','abysmal','abysmally','abyss','accidental','accost','accursed','accusation','accusations','accuse','accuses','accusing','accusingly','acerbate','acerbic','acerbically','ache','ached','aches','achey','aching','acrid','acridly','acridness','acrimonious','acrimoniously','acrimony','adamant','adamantly','addict','addicted','addicting','addicts','admonish','admonisher','admonishingly','admonishment','admonition','adulterate','adulterated','adulteration','adulterier','adversarial','adversary','adverse','adversity','afflict','affliction','afflictive','affront','afraid','aggravate','aggravating','aggravation','aggression','aggressive','aggressiveness','aggressor','aggrieve','aggrieved','aggrivation','aghast','agonies','agonize','agonizing','agonizingly','agony','aground','ail','ailing','ailment','aimless','alarm','alarmed','alarming','alarmingly','alienate','alienated','alienation','allegation','allegations','allege','allergic','allergies','allergy','aloof','altercation','ambiguity','ambiguous','ambivalence','ambivalent','ambush','amiss','amputate','anarchism','anarchist','anarchistic','anarchy','anemic','anger','angrily','angriness','angry','anguish','animosity','annihilate','annihilation','annoy','annoyance','annoyances','annoyed','annoying','annoyingly','annoys','anomalous','anomaly','antagonism','antagonist','antagonistic','antagonize','anti-','anti-american','anti-israeli','anti-occupation','anti-proliferation','anti-semites','anti-social','anti-us','anti-white','antipathy','antiquated','antithetical','anxieties','anxiety','anxious','anxiously','anxiousness','apathetic','apathetically','apathy','apocalypse','apocalyptic','apologist','apologists','appal','appall','appalled','appalling','appallingly','apprehension','apprehensions','apprehensive','apprehensively','arbitrary','arcane','archaic','arduous','arduously','argumentative','arrogance','arrogant','arrogantly','ashamed','asinine','asininely','asinininity','askance','asperse','aspersion','aspersions','assail','assassin','assassinate','assault','assult','astray','asunder','atrocious','atrocities','atrocity','atrophy','attack','attacks','audacious','audaciously','audaciousness','audacity','audiciously','austere','authoritarian','autocrat','autocratic','avalanche','avarice','avaricious','avariciously','avenge','averse','aversion','aweful','awful','awfully','awfulness','awkward','awkwardness','ax','babble','back-logged','back-wood','back-woods','backache','backaches','backaching','backbite','backbiting','backward','backwardness','backwood','backwoods','bad','badly','baffle','baffled','bafflement','baffling','bait','balk','banal','banalize','bane','banish','banishment','bankrupt','barbarian','barbaric','barbarically','barbarity','barbarous','barbarously','barren','baseless','bash','bashed','bashful','bashing','bastard','bastards','battered','battering','batty','bearish','beastly','bedlam','bedlamite','befoul','beg','beggar','beggarly','begging','beguile','belabor','belated','beleaguer','belie','belittle','belittled','belittling','bellicose','belligerence','belligerent','belligerently','bemoan','bemoaning','bemused','bent','berate','bereave','bereavement','bereft','berserk','beseech','beset','besiege','besmirch','bestial','betray','betrayal','betrayals','betrayer','betraying','betrays','bewail','beware','bewilder','bewildered','bewildering','bewilderingly','bewilderment','bewitch','bias','biased','biases','bicker','bickering','bid-rigging','bigotries','bigotry','bitch','bitchy','biting','bitingly','bitter','bitterly','bitterness','bizarre','blab','blabber','blackmail','blah','blame','blameworthy','bland','blandish','blaspheme','blasphemous','blasphemy','blasted','blatant','blatantly','blather','bleak','bleakly','bleakness','bleed','bleeding','bleeds','blemish','blind','blinding','blindingly','blindside','blister','blistering','bloated','blockage','blockhead','bloodshed','bloodthirsty','bloody','blotchy','blow','blunder','blundering','blunders','blunt','blur','bluring','blurred','blurring','blurry','blurs','blurt','boastful','boggle','bogus','boil','boiling','boisterous','bomb','bombard','bombardment','bombastic','bondage','bonkers','bore','bored','boredom','bores','boring','botch','bother','bothered','bothering','bothers','bothersome','bowdlerize','boycott','braggart','bragger','brainless','brainwash','brash','brashly','brashness','brat','bravado','brazen','brazenly','brazenness','breach','break','break-up','break-ups','breakdown','breaking','breaks','breakup','breakups','bribery','brimstone','bristle','brittle','broke','broken','broken-hearted','brood','browbeat','bruise','bruised','bruises','bruising','brusque','brutal','brutalising','brutalities','brutality','brutalize','brutalizing','brutally','brute','brutish','bs','buckle','bug','bugging','buggy','bugs','bulkier','bulkiness','bulky','bulkyness','bull****','bull----','bullies','bullshit','bullshyt','bully','bullying','bullyingly','bum','bump','bumped','bumping','bumpping','bumps','bumpy','bungle','bungler','bungling','bunk','burden','burdensome','burdensomely','burn','burned','burning','burns','bust','busts','busybody','butcher','butchery','buzzing','byzantine','cackle','calamities','calamitous','calamitously','calamity','callous','calumniate','calumniation','calumnies','calumnious','calumniously','calumny','cancer','cancerous','cannibal','cannibalize','capitulate','capricious','capriciously','capriciousness','capsize','careless','carelessness','caricature','carnage','carp','cartoonish','cash-strapped','castigate','castrated','casualty','cataclysm','cataclysmal','cataclysmic','cataclysmically','catastrophe','catastrophes','catastrophic','catastrophically','catastrophies','caustic','caustically','cautionary','cave','censure','chafe','chaff','chagrin','challenging','chaos','chaotic','chasten','chastise','chastisement','chatter','chatterbox','cheap','cheapen','cheaply','cheat','cheated','cheater','cheating','cheats','checkered','cheerless','cheesy','chide','childish','chill','chilly','chintzy','choke','choleric','choppy','chore','chronic','chunky','clamor','clamorous','clash','cliche','cliched','clique','clog','clogged','clogs','cloud','clouding','cloudy','clueless','clumsy','clunky','coarse','cocky','coerce','coercion','coercive','cold','coldly','collapse','collude','collusion','combative','combust','comical','commiserate','commonplace','commotion','commotions','complacent','complain','complained','complaining','complains','complaint','complaints','complex','complicated','complication','complicit','compulsion','compulsive','concede','conceded','conceit','conceited','concen','concens','concern','concerned','concerns','concession','concessions','condemn','condemnable','condemnation','condemned','condemns','condescend','condescending','condescendingly','condescension','confess','confession','confessions','confined','conflict','conflicted','conflicting','conflicts','confound','confounded','confounding','confront','confrontation','confrontational','confuse','confused','confuses','confusing','confusion','confusions','congested','congestion','cons','conscons','conservative','conspicuous','conspicuously','conspiracies','conspiracy','conspirator','conspiratorial','conspire','consternation','contagious','contaminate','contaminated','contaminates','contaminating','contamination','contempt','contemptible','contemptuous','contemptuously','contend','contention','contentious','contort','contortions','contradict','contradiction','contradictory','contrariness','contravene','contrive','contrived','controversial','controversy','convoluted','corrode','corrosion','corrosions','corrosive','corrupt','corrupted','corrupting','corruption','corrupts','corruptted','costlier','costly','counter-productive','counterproductive','coupists','covetous','coward','cowardly','crabby','crack','cracked','cracks','craftily','craftly','crafty','cramp','cramped','cramping','cranky','crap','crappy','craps','crash','crashed','crashes','crashing','crass','craven','cravenly','craze','crazily','craziness','crazy','creak','creaking','creaks','credulous','creep','creeping','creeps','creepy','crept','crime','criminal','cringe','cringed','cringes','cripple','crippled','cripples','crippling','crisis','critic','critical','criticism','criticisms','criticize','criticized','criticizing','critics','cronyism','crook','crooked','crooks','crowded','crowdedness','crude','cruel','crueler','cruelest','cruelly','cruelness','cruelties','cruelty','crumble','crumbling','crummy','crumple','crumpled','crumples','crush','crushed','crushing','cry','culpable','culprit','cumbersome','cunt','cunts','cuplrit','curse','cursed','curses','curt','cuss','cussed','cutthroat','cynical','cynicism','d*mn','damage','damaged','damages','damaging','damn','damnable','damnably','damnation','damned','damning','damper','danger','dangerous','dangerousness','dark','darken','darkened','darker','darkness','dastard','dastardly','daunt','daunting','dauntingly','dawdle','daze','dazed','dead','deadbeat','deadlock','deadly','deadweight','deaf','dearth','death','debacle','debase','debasement','debaser','debatable','debauch','debaucher','debauchery','debilitate','debilitating','debility','debt','debts','decadence','decadent','decay','decayed','deceit','deceitful','deceitfully','deceitfulness','deceive','deceiver','deceivers','deceiving','deception','deceptive','deceptively','declaim','decline','declines','declining','decrement','decrepit','decrepitude','decry','defamation','defamations','defamatory','defame','defect','defective','defects','defensive','defiance','defiant','defiantly','deficiencies','deficiency','deficient','defile','defiler','deform','deformed','defrauding','defunct','defy','degenerate','degenerately','degeneration','degradation','degrade','degrading','degradingly','dehumanization','dehumanize','deign','deject','dejected','dejectedly','dejection','delay','delayed','delaying','delays','delinquency','delinquent','delirious','delirium','delude','deluded','deluge','delusion','delusional','delusions','demean','demeaning','demise','demolish','demolisher','demon','demonic','demonize','demonized','demonizes','demonizing','demoralize','demoralizing','demoralizingly','denial','denied','denies','denigrate','denounce','dense','dent','dented','dents','denunciate','denunciation','denunciations','deny','denying','deplete','deplorable','deplorably','deplore','deploring','deploringly','deprave','depraved','depravedly','deprecate','depress','depressed','depressing','depressingly','depression','depressions','deprive','deprived','deride','derision','derisive','derisively','derisiveness','derogatory','desecrate','desert','desertion','desiccate','desiccated','desititute','desolate','desolately','desolation','despair','despairing','despairingly','desperate','desperately','desperation','despicable','despicably','despise','despised','despoil','despoiler','despondence','despondency','despondent','despondently','despot','despotic','despotism','destabilisation','destains','destitute','destitution','destroy','destroyer','destruction','destructive','desultory','deter','deteriorate','deteriorating','deterioration','deterrent','detest','detestable','detestably','detested','detesting','detests','detract','detracted','detracting','detraction','detracts','detriment','detrimental','devastate','devastated','devastates','devastating','devastatingly','devastation','deviate','deviation','devil','devilish','devilishly','devilment','devilry','devious','deviously','deviousness','devoid','diabolic','diabolical','diabolically','diametrically','diappointed','diatribe','diatribes','dick','dictator','dictatorial','die','die-hard','died','dies','difficult','difficulties','difficulty','diffidence','dilapidated','dilemma','dilly-dally','dim','dimmer','din','ding','dings','dinky','dire','direly','direness','dirt','dirtbag','dirtbags','dirts','dirty','disable','disabled','disaccord','disadvantage','disadvantaged','disadvantageous','disadvantages','disaffect','disaffected','disaffirm','disagree','disagreeable','disagreeably','disagreed','disagreeing','disagreement','disagrees','disallow','disapointed','disapointing','disapointment','disappoint','disappointed','disappointing','disappointingly','disappointment','disappointments','disappoints','disapprobation','disapproval','disapprove','disapproving','disarm','disarray','disaster','disasterous','disastrous','disastrously','disavow','disavowal','disbelief','disbelieve','disbeliever','disclaim','discombobulate','discomfit','discomfititure','discomfort','discompose','disconcert','disconcerted','disconcerting','disconcertingly','disconsolate','disconsolately','disconsolation','discontent','discontented','discontentedly','discontinued','discontinuity','discontinuous','discord','discordance','discordant','discountenance','discourage','discouragement','discouraging','discouragingly','discourteous','discourteously','discoutinous','discredit','discrepant','discriminate','discrimination','discriminatory','disdain','disdained','disdainful','disdainfully','disfavor','disgrace','disgraced','disgraceful','disgracefully','disgruntle','disgruntled','disgust','disgusted','disgustedly','disgustful','disgustfully','disgusting','disgustingly','dishearten','disheartening','dishearteningly','dishonest','dishonestly','dishonesty','dishonor','dishonorable','dishonorablely','disillusion','disillusioned','disillusionment','disillusions','disinclination','disinclined','disingenuous','disingenuously','disintegrate','disintegrated','disintegrates','disintegration','disinterest','disinterested','dislike','disliked','dislikes','disliking','dislocated','disloyal','disloyalty','dismal','dismally','dismalness','dismay','dismayed','dismaying','dismayingly','dismissive','dismissively','disobedience','disobedient','disobey','disoobedient','disorder','disordered','disorderly','disorganized','disorient','disoriented','disown','disparage','disparaging','disparagingly','dispensable','dispirit','dispirited','dispiritedly','dispiriting','displace','displaced','displease','displeased','displeasing','displeasure','disproportionate','disprove','disputable','dispute','disputed','disquiet','disquieting','disquietingly','disquietude','disregard','disregardful','disreputable','disrepute','disrespect','disrespectable','disrespectablity','disrespectful','disrespectfully','disrespectfulness','disrespecting','disrupt','disruption','disruptive','diss','dissapointed','dissappointed','dissappointing','dissatisfaction','dissatisfactory','dissatisfied','dissatisfies','dissatisfy','dissatisfying','dissed','dissemble','dissembler','dissension','dissent','dissenter','dissention','disservice','disses','dissidence','dissident','dissidents','dissing','dissocial','dissolute','dissolution','dissonance','dissonant','dissonantly','dissuade','dissuasive','distains','distaste','distasteful','distastefully','distort','distorted','distortion','distorts','distract','distracting','distraction','distraught','distraughtly','distraughtness','distress','distressed','distressing','distressingly','distrust','distrustful','distrusting','disturb','disturbance','disturbed','disturbing','disturbingly','disunity','disvalue','divergent','divisive','divisively','divisiveness','dizzing','dizzingly','dizzy','doddering','dodgey','dogged','doggedly','dogmatic','doldrums','domineer','domineering','donside','doom','doomed','doomsday','dope','doubt','doubtful','doubtfully','doubts','douchbag','douchebag','douchebags','downbeat','downcast','downer','downfall','downfallen','downgrade','downhearted','downheartedly','downhill','downside','downsides','downturn','downturns','drab','draconian','draconic','drag','dragged','dragging','dragoon','drags','drain','drained','draining','drains','drastic','drastically','drawback','drawbacks','dread','dreadful','dreadfully','dreadfulness','dreary','dripped','dripping','drippy','drips','drones','droop','droops','drop-out','drop-outs','dropout','dropouts','drought','drowning','drunk','drunkard','drunken','dubious','dubiously','dubitable','dud','dull','dullard','dumb','dumbfound','dump','dumped','dumping','dumps','dunce','dungeon','dungeons','dupe','dust','dusty','dwindling','dying','earsplitting','eccentric','eccentricity','effigy','effrontery','egocentric','egomania','egotism','egotistical','egotistically','egregious','egregiously','election-rigger','elimination','emaciated','emasculate','embarrass','embarrassing','embarrassingly','embarrassment','embattled','embroil','embroiled','embroilment','emergency','emphatic','emphatically','emptiness','encroach','encroachment','endanger','enemies','enemy','enervate','enfeeble','enflame','engulf','enjoin','enmity','enrage','enraged','enraging','enslave','entangle','entanglement','entrap','entrapment','envious','enviously','enviousness','epidemic','equivocal','erase','erode','erodes','erosion','err','errant','erratic','erratically','erroneous','erroneously','error','errors','eruptions','escapade','eschew','estranged','evade','evasion','evasive','evil','evildoer','evils','eviscerate','exacerbate','exagerate','exagerated','exagerates','exaggerate','exaggeration','exasperate','exasperated','exasperating','exasperatingly','exasperation','excessive','excessively','exclusion','excoriate','excruciating','excruciatingly','excuse','excuses','execrate','exhaust','exhausted','exhaustion','exhausts','exhorbitant','exhort','exile','exorbitant','exorbitantance','exorbitantly','expel','expensive','expire','expired','explode','exploit','exploitation','explosive','expropriate','expropriation','expulse','expunge','exterminate','extermination','extinguish','extort','extortion','extraneous','extravagance','extravagant','extravagantly','extremism','extremist','extremists','eyesore','f**k','fabricate','fabrication','facetious','facetiously','fail','failed','failing','fails','failure','failures','faint','fainthearted','faithless','fake','fall','fallacies','fallacious','fallaciously','fallaciousness','fallacy','fallen','falling','fallout','falls','false','falsehood','falsely','falsify','falter','faltered','famine','famished','fanatic','fanatical','fanatically','fanaticism','fanatics','fanciful','far-fetched','farce','farcical','farcical-yet-provocative','farcically','farfetched','fascism','fascist','fastidious','fastidiously','fastuous','fat','fat-cat','fat-cats','fatal','fatalistic','fatalistically','fatally','fatcat','fatcats','fateful','fatefully','fathomless','fatigue','fatigued','fatique','fatty','fatuity','fatuous','fatuously','fault','faults','faulty','fawningly','faze','fear','fearful','fearfully','fears','fearsome','feckless','feeble','feeblely','feebleminded','feign','feint','fell','felon','felonious','ferociously','ferocity','fetid','fever','feverish','fevers','fiasco','fib','fibber','fickle','fiction','fictional','fictitious','fidget','fidgety','fiend','fiendish','fierce','figurehead','filth','filthy','finagle','finicky','fissures','fist','flabbergast','flabbergasted','flagging','flagrant','flagrantly','flair','flairs','flak','flake','flakey','flakieness','flaking','flaky','flare','flares','flareup','flareups','flat-out','flaunt','flaw','flawed','flaws','flee','fleed','fleeing','fleer','flees','fleeting','flicering','flicker','flickering','flickers','flighty','flimflam','flimsy','flirt','flirty','floored','flounder','floundering','flout','fluster','foe','fool','fooled','foolhardy','foolish','foolishly','foolishness','forbid','forbidden','forbidding','forceful','foreboding','forebodingly','forfeit','forged','forgetful','forgetfully','forgetfulness','forlorn','forlornly','forsake','forsaken','forswear','foul','foully','foulness','fractious','fractiously','fracture','fragile','fragmented','frail','frantic','frantically','franticly','fraud','fraudulent','fraught','frazzle','frazzled','freak','freaking','freakish','freakishly','freaks','freeze','freezes','freezing','frenetic','frenetically','frenzied','frenzy','fret','fretful','frets','friction','frictions','fried','friggin','frigging','fright','frighten','frightening','frighteningly','frightful','frightfully','frigid','frost','frown','froze','frozen','fruitless','fruitlessly','frustrate','frustrated','frustrates','frustrating','frustratingly','frustration','frustrations','fuck','fucking','fudge','fugitive','full-blown','fulminate','fumble','fume','fumes','fundamentalism','funky','funnily','funny','furious','furiously','furor','fury','fuss','fussy','fustigate','fusty','futile','futilely','futility','fuzzy','gabble','gaff','gaffe','gainsay','gainsayer','gall','galling','gallingly','galls','gangster','gape','garbage','garish','gasp','gauche','gaudy','gawk','gawky','geezer','genocide','get-rich','ghastly','ghetto','ghosting','gibber','gibberish','gibe','giddy','gimmick','gimmicked','gimmicking','gimmicks','gimmicky','glare','glaringly','glib','glibly','glitch','glitches','gloatingly','gloom','gloomy','glower','glum','glut','gnawing','goad','goading','god-awful','goof','goofy','goon','gossip','graceless','gracelessly','graft','grainy','grapple','grate','grating','gravely','greasy','greed','greedy','grief','grievance','grievances','grieve','grieving','grievous','grievously','grim','grimace','grind','gripe','gripes','grisly','gritty','gross','grossly','grotesque','grouch','grouchy','groundless','grouse','growl','grudge','grudges','grudging','grudgingly','gruesome','gruesomely','gruff','grumble','grumpier','grumpiest','grumpily','grumpish','grumpy','guile','guilt','guiltily','guilty','gullible','gutless','gutter','hack','hacks','haggard','haggle','hairloss','halfhearted','halfheartedly','hallucinate','hallucination','hamper','hampered','handicapped','hang','hangs','haphazard','hapless','harangue','harass','harassed','harasses','harassment','harboring','harbors','hard','hard-hit','hard-line','hard-liner','hardball','harden','hardened','hardheaded','hardhearted','hardliner','hardliners','hardship','hardships','harm','harmed','harmful','harms','harpy','harridan','harried','harrow','harsh','harshly','hasseling','hassle','hassled','hassles','haste','hastily','hasty','hate','hated','hateful','hatefully','hatefulness','hater','haters','hates','hating','hatred','haughtily','haughty','haunt','haunting','havoc','hawkish','haywire','hazard','hazardous','haze','hazy','head-aches','headache','headaches','heartbreaker','heartbreaking','heartbreakingly','heartless','heathen','heavy-handed','heavyhearted','heck','heckle','heckled','heckles','hectic','hedge','hedonistic','heedless','hefty','hegemonism','hegemonistic','hegemony','heinous','hell','hell-bent','hellion','hells','helpless','helplessly','helplessness','heresy','heretic','heretical','hesitant','hestitant','hideous','hideously','hideousness','high-priced','hiliarious','hinder','hindrance','hiss','hissed','hissing','ho-hum','hoard','hoax','hobble','hogs','hollow','hoodium','hoodwink','hooligan','hopeless','hopelessly','hopelessness','horde','horrendous','horrendously','horrible','horrid','horrific','horrified','horrifies','horrify','horrifying','horrifys','hostage','hostile','hostilities','hostility','hotbeds','hothead','hotheaded','hothouse','hubris','huckster','hum','humid','humiliate','humiliating','humiliation','humming','hung','hurt','hurted','hurtful','hurting','hurts','hustler','hype','hypocricy','hypocrisy','hypocrite','hypocrites','hypocritical','hypocritically','hysteria','hysteric','hysterical','hysterically','hysterics','idiocies','idiocy','idiot','idiotic','idiotically','idiots','idle','ignoble','ignominious','ignominiously','ignominy','ignorance','ignorant','ignore','ill-advised','ill-conceived','ill-defined','ill-designed','ill-fated','ill-favored','ill-formed','ill-mannered','ill-natured','ill-sorted','ill-tempered','ill-treated','ill-treatment','ill-usage','ill-used','illegal','illegally','illegitimate','illicit','illiterate','illness','illogic','illogical','illogically','illusion','illusions','illusory','imaginary','imbalance','imbecile','imbroglio','immaterial','immature','imminence','imminently','immobilized','immoderate','immoderately','immodest','immoral','immorality','immorally','immovable','impair','impaired','impasse','impatience','impatient','impatiently','impeach','impedance','impede','impediment','impending','impenitent','imperfect','imperfection','imperfections','imperfectly','imperialist','imperil','imperious','imperiously','impermissible','impersonal','impertinent','impetuous','impetuously','impiety','impinge','impious','implacable','implausible','implausibly','implicate','implication','implode','impolite','impolitely','impolitic','importunate','importune','impose','imposers','imposing','imposition','impossible','impossiblity','impossibly','impotent','impoverish','impoverished','impractical','imprecate','imprecise','imprecisely','imprecision','imprison','imprisonment','improbability','improbable','improbably','improper','improperly','impropriety','imprudence','imprudent','impudence','impudent','impudently','impugn','impulsive','impulsively','impunity','impure','impurity','inability','inaccuracies','inaccuracy','inaccurate','inaccurately','inaction','inactive','inadequacy','inadequate','inadequately','inadverent','inadverently','inadvisable','inadvisably','inane','inanely','inappropriate','inappropriately','inapt','inaptitude','inarticulate','inattentive','inaudible','incapable','incapably','incautious','incendiary','incense','incessant','incessantly','incite','incitement','incivility','inclement','incognizant','incoherence','incoherent','incoherently','incommensurate','incomparable','incomparably','incompatability','incompatibility','incompatible','incompetence','incompetent','incompetently','incomplete','incompliant','incomprehensible','incomprehension','inconceivable','inconceivably','incongruous','incongruously','inconsequent','inconsequential','inconsequentially','inconsequently','inconsiderate','inconsiderately','inconsistence','inconsistencies','inconsistency','inconsistent','inconsolable','inconsolably','inconstant','inconvenience','inconveniently','incorrect','incorrectly','incorrigible','incorrigibly','incredulous','incredulously','inculcate','indecency','indecent','indecently','indecision','indecisive','indecisively','indecorum','indefensible','indelicate','indeterminable','indeterminably','indeterminate','indifference','indifferent','indigent','indignant','indignantly','indignation','indignity','indiscernible','indiscreet','indiscreetly','indiscretion','indiscriminate','indiscriminately','indiscriminating','indistinguishable','indoctrinate','indoctrination','indolent','indulge','ineffective','ineffectively','ineffectiveness','ineffectual','ineffectually','ineffectualness','inefficacious','inefficacy','inefficiency','inefficient','inefficiently','inelegance','inelegant','ineligible','ineloquent','ineloquently','inept','ineptitude','ineptly','inequalities','inequality','inequitable','inequitably','inequities','inescapable','inescapably','inessential','inevitable','inevitably','inexcusable','inexcusably','inexorable','inexorably','inexperience','inexperienced','inexpert','inexpertly','inexpiable','inexplainable','inextricable','inextricably','infamous','infamously','infamy','infected','infection','infections','inferior','inferiority','infernal','infest','infested','infidel','infidels','infiltrator','infiltrators','infirm','inflame','inflammation','inflammatory','inflammed','inflated','inflationary','inflexible','inflict','infraction','infringe','infringement','infringements','infuriate','infuriated','infuriating','infuriatingly','inglorious','ingrate','ingratitude','inhibit','inhibition','inhospitable','inhospitality','inhuman','inhumane','inhumanity','inimical','inimically','iniquitous','iniquity','injudicious','injure','injurious','injury','injustice','injustices','innuendo','inoperable','inopportune','inordinate','inordinately','insane','insanely','insanity','insatiable','insecure','insecurity','insensible','insensitive','insensitively','insensitivity','insidious','insidiously','insignificance','insignificant','insignificantly','insincere','insincerely','insincerity','insinuate','insinuating','insinuation','insociable','insolence','insolent','insolently','insolvent','insouciance','instability','instable','instigate','instigator','instigators','insubordinate','insubstantial','insubstantially','insufferable','insufferably','insufficiency','insufficient','insufficiently','insular','insult','insulted','insulting','insultingly','insults','insupportable','insupportably','insurmountable','insurmountably','insurrection','intefere','inteferes','intense','interfere','interference','interferes','intermittent','interrupt','interruption','interruptions','intimidate','intimidating','intimidatingly','intimidation','intolerable','intolerablely','intolerance','intoxicate','intractable','intransigence','intransigent','intrude','intrusion','intrusive','inundate','inundated','invader','invalid','invalidate','invalidity','invasive','invective','inveigle','invidious','invidiously','invidiousness','invisible','involuntarily','involuntary','irascible','irate','irately','ire','irk','irked','irking','irks','irksome','irksomely','irksomeness','irksomenesses','ironic','ironical','ironically','ironies','irony','irragularity','irrational','irrationalities','irrationality','irrationally','irrationals','irreconcilable','irrecoverable','irrecoverableness','irrecoverablenesses','irrecoverably','irredeemable','irredeemably','irreformable','irregular','irregularity','irrelevance','irrelevant','irreparable','irreplacible','irrepressible','irresolute','irresolvable','irresponsible','irresponsibly','irretating','irretrievable','irreversible','irritable','irritably','irritant','irritate','irritated','irritating','irritation','irritations','isolate','isolated','isolation','issue','issues','itch','itching','itchy','jabber','jaded','jagged','jam','jarring','jaundiced','jealous','jealously','jealousness','jealousy','jeer','jeering','jeeringly','jeers','jeopardize','jeopardy','jerk','jerky','jitter','jitters','jittery','job-killing','jobless','joke','joker','jolt','judder','juddering','judders','jumpy','junk','junky','junkyard','jutter','jutters','kaput','kill','killed','killer','killing','killjoy','kills','knave','knife','knock','knotted','kook','kooky','lack','lackadaisical','lacked','lackey','lackeys','lacking','lackluster','lacks','laconic','lag','lagged','lagging','laggy','lags','laid-off','lambast','lambaste','lame','lame-duck','lament','lamentable','lamentably','languid','languish','languor','languorous','languorously','lanky','lapse','lapsed','lapses','lascivious','last-ditch','latency','laughable','laughably','laughingstock','lawbreaker','lawbreaking','lawless','lawlessness','layoff','layoff-happy','lazy','leak','leakage','leakages','leaking','leaks','leaky','lech','lecher','lecherous','lechery','leech','leer','leery','left-leaning','lemon','lengthy','less-developed','lesser-known','letch','lethal','lethargic','lethargy','lewd','lewdly','lewdness','liability','liable','liar','liars','licentious','licentiously','licentiousness','lie','lied','lier','lies','life-threatening','lifeless','limit','limitation','limitations','limited','limits','limp','listless','litigious','little-known','livid','lividly','loath','loathe','loathing','loathly','loathsome','loathsomely','lone','loneliness','lonely','loner','lonesome','long-time','long-winded','longing','longingly','loophole','loopholes','loose','loot','lorn','lose','loser','losers','loses','losing','loss','losses','lost','loud','louder','lousy','loveless','lovelorn','low-rated','lowly','ludicrous','ludicrously','lugubrious','lukewarm','lull','lumpy','lunatic','lunaticism','lurch','lure','lurid','lurk','lurking','lying','macabre','mad','madden','maddening','maddeningly','madder','madly','madman','madness','maladjusted','maladjustment','malady','malaise','malcontent','malcontented','maledict','malevolence','malevolent','malevolently','malice','malicious','maliciously','maliciousness','malign','malignant','malodorous','maltreatment','mangle','mangled','mangles','mangling','mania','maniac','maniacal','manic','manipulate','manipulation','manipulative','manipulators','mar','marginal','marginally','martyrdom','martyrdom-seeking','mashed','massacre','massacres','matte','mawkish','mawkishly','mawkishness','meager','meaningless','meanness','measly','meddle','meddlesome','mediocre','mediocrity','melancholy','melodramatic','melodramatically','meltdown','menace','menacing','menacingly','mendacious','mendacity','menial','merciless','mercilessly','mess','messed','messes','messing','messy','midget','miff','militancy','mindless','mindlessly','mirage','mire','misalign','misaligned','misaligns','misapprehend','misbecome','misbecoming','misbegotten','misbehave','misbehavior','miscalculate','miscalculation','miscellaneous','mischief','mischievous','mischievously','misconception','misconceptions','miscreant','miscreants','misdirection','miser','miserable','miserableness','miserably','miseries','miserly','misery','misfit','misfortune','misgiving','misgivings','misguidance','misguide','misguided','mishandle','mishap','misinform','misinformed','misinterpret','misjudge','misjudgment','mislead','misleading','misleadingly','mislike','mismanage','mispronounce','mispronounced','mispronounces','misread','misreading','misrepresent','misrepresentation','miss','missed','misses','misstatement','mist','mistake','mistaken','mistakenly','mistakes','mistified','mistress','mistrust','mistrustful','mistrustfully','mists','misunderstand','misunderstanding','misunderstandings','misunderstood','misuse','moan','mobster','mock','mocked','mockeries','mockery','mocking','mockingly','mocks','molest','molestation','monotonous','monotony','monster','monstrosities','monstrosity','monstrous','monstrously','moody','moot','mope','morbid','morbidly','mordant','mordantly','moribund','moron','moronic','morons','mortification','mortified','mortify','mortifying','motionless','motley','mourn','mourner','mournful','mournfully','muddle','muddy','mudslinger','mudslinging','mulish','multi-polarization','mundane','murder','murderer','murderous','murderously','murky','muscle-flexing','mushy','musty','mysterious','mysteriously','mystery','mystify','myth','nag','nagging','naive','naively','narrower','nastily','nastiness','nasty','naughty','nauseate','nauseates','nauseating','nauseatingly','na�ve','nebulous','nebulously','needless','needlessly','needy','nefarious','nefariously','negate','negation','negative','negatives','negativity','neglect','neglected','negligence','negligent','nemesis','nepotism','nervous','nervously','nervousness','nettle','nettlesome','neurotic','neurotically','niggle','niggles','nightmare','nightmarish','nightmarishly','nitpick','nitpicking','noise','noises','noisier','noisy','non-confidence','nonexistent','nonresponsive','nonsense','nosey','notoriety','notorious','notoriously','noxious','nuisance','numb','obese','object','objection','objectionable','objections','oblique','obliterate','obliterated','oblivious','obnoxious','obnoxiously','obscene','obscenely','obscenity','obscure','obscured','obscures','obscurity','obsess','obsessive','obsessively','obsessiveness','obsolete','obstacle','obstinate','obstinately','obstruct','obstructed','obstructing','obstruction','obstructs','obtrusive','obtuse','occlude','occluded','occludes','occluding','odd','odder','oddest','oddities','oddity','oddly','odor','offence','offend','offender','offending','offenses','offensive','offensively','offensiveness','officious','ominous','ominously','omission','omit','one-sided','onerous','onerously','onslaught','opinionated','opponent','opportunistic','oppose','opposition','oppositions','oppress','oppression','oppressive','oppressively','oppressiveness','oppressors','ordeal','orphan','ostracize','outbreak','outburst','outbursts','outcast','outcry','outlaw','outmoded','outrage','outraged','outrageous','outrageously','outrageousness','outrages','outsider','over-acted','over-awe','over-balanced','over-hyped','over-priced','over-valuation','overact','overacted','overawe','overbalance','overbalanced','overbearing','overbearingly','overblown','overdo','overdone','overdue','overemphasize','overheat','overkill','overloaded','overlook','overpaid','overpayed','overplay','overpower','overpriced','overrated','overreach','overrun','overshadow','oversight','oversights','oversimplification','oversimplified','oversimplify','oversize','overstate','overstated','overstatement','overstatements','overstates','overtaxed','overthrow','overthrows','overturn','overweight','overwhelm','overwhelmed','overwhelming','overwhelmingly','overwhelms','overzealous','overzealously','overzelous','pain','painful','painfull','painfully','pains','pale','pales','paltry','pan','pandemonium','pander','pandering','panders','panic','panick','panicked','panicking','panicky','paradoxical','paradoxically','paralize','paralyzed','paranoia','paranoid','parasite','pariah','parody','partiality','partisan','partisans','passe','passive','passiveness','pathetic','pathetically','patronize','paucity','pauper','paupers','payback','peculiar','peculiarly','pedantic','peeled','peeve','peeved','peevish','peevishly','penalize','penalty','perfidious','perfidity','perfunctory','peril','perilous','perilously','perish','pernicious','perplex','perplexed','perplexing','perplexity','persecute','persecution','pertinacious','pertinaciously','pertinacity','perturb','perturbed','pervasive','perverse','perversely','perversion','perversity','pervert','perverted','perverts','pessimism','pessimistic','pessimistically','pest','pestilent','petrified','petrify','pettifog','petty','phobia','phobic','phony','picket','picketed','picketing','pickets','picky','pig','pigs','pillage','pillory','pimple','pinch','pique','pitiable','pitiful','pitifully','pitiless','pitilessly','pittance','pity','plagiarize','plague','plasticky','plaything','plea','pleas','plebeian','plight','plot','plotters','ploy','plunder','plunderer','pointless','pointlessly','poison','poisonous','poisonously','pokey','poky','polarisation','polemize','pollute','polluter','polluters','polution','pompous','poor','poorer','poorest','poorly','posturing','pout','poverty','powerless','prate','pratfall','prattle','precarious','precariously','precipitate','precipitous','predatory','predicament','prejudge','prejudice','prejudices','prejudicial','premeditated','preoccupy','preposterous','preposterously','presumptuous','presumptuously','pretence','pretend','pretense','pretentious','pretentiously','prevaricate','pricey','pricier','prick','prickle','prickles','prideful','prik','primitive','prison','prisoner','problem','problematic','problems','procrastinate','procrastinates','procrastination','profane','profanity','prohibit','prohibitive','prohibitively','propaganda','propagandize','proprietary','prosecute','protest','protested','protesting','protests','protracted','provocation','provocative','provoke','pry','pugnacious','pugnaciously','pugnacity','punch','punish','punishable','punitive','punk','puny','puppet','puppets','puzzled','puzzlement','puzzling','quack','qualm','qualms','quandary','quarrel','quarrellous','quarrellously','quarrels','quarrelsome','quash','queer','questionable','quibble','quibbles','quitter','rabid','racism','racist','racists','racy','radical','radicalization','radically','radicals','rage','ragged','raging','rail','raked','rampage','rampant','ramshackle','rancor','randomly','rankle','rant','ranted','ranting','rantingly','rants','rape','raped','raping','rascal','rascals','rash','rattle','rattled','rattles','ravage','raving','reactionary','rebellious','rebuff','rebuke','recalcitrant','recant','recession','recessionary','reckless','recklessly','recklessness','recoil','recourses','redundancy','redundant','refusal','refuse','refused','refuses','refusing','refutation','refute','refuted','refutes','refuting','regress','regression','regressive','regret','regreted','regretful','regretfully','regrets','regrettable','regrettably','regretted','reject','rejected','rejecting','rejection','rejects','relapse','relentless','relentlessly','relentlessness','reluctance','reluctant','reluctantly','remorse','remorseful','remorsefully','remorseless','remorselessly','remorselessness','renounce','renunciation','repel','repetitive','reprehensible','reprehensibly','reprehension','reprehensive','repress','repression','repressive','reprimand','reproach','reproachful','reprove','reprovingly','repudiate','repudiation','repugn','repugnance','repugnant','repugnantly','repulse','repulsed','repulsing','repulsive','repulsively','repulsiveness','resent','resentful','resentment','resignation','resigned','resistance','restless','restlessness','restrict','restricted','restriction','restrictive','resurgent','retaliate','retaliatory','retard','retarded','retardedness','retards','reticent','retract','retreat','retreated','revenge','revengeful','revengefully','revert','revile','reviled','revoke','revolt','revolting','revoltingly','revulsion','revulsive','rhapsodize','rhetoric','rhetorical','ricer','ridicule','ridicules','ridiculous','ridiculously','rife','rift','rifts','rigid','rigidity','rigidness','rile','riled','rip','rip-off','ripoff','ripped','risk','risks','risky','rival','rivalry','roadblocks','rocky','rogue','rollercoaster','rot','rotten','rough','rremediable','rubbish','rude','rue','ruffian','ruffle','ruin','ruined','ruining','ruinous','ruins','rumbling','rumor','rumors','rumours','rumple','run-down','runaway','rupture','rust','rusts','rusty','rut','ruthless','ruthlessly','ruthlessness','ruts','sabotage','sack','sacrificed','sad','sadden','sadly','sadness','sag','sagged','sagging','saggy','sags','salacious','sanctimonious','sap','sarcasm','sarcastic','sarcastically','sardonic','sardonically','sass','satirical','satirize','savage','savaged','savagery','savages','scaly','scam','scams','scandal','scandalize','scandalized','scandalous','scandalously','scandals','scandel','scandels','scant','scapegoat','scar','scarce','scarcely','scarcity','scare','scared','scarier','scariest','scarily','scarred','scars','scary','scathing','scathingly','sceptical','scoff','scoffingly','scold','scolded','scolding','scoldingly','scorching','scorchingly','scorn','scornful','scornfully','scoundrel','scourge','scowl','scramble','scrambled','scrambles','scrambling','scrap','scratch','scratched','scratches','scratchy','scream','screech','screw-up','screwed','screwed-up','screwy','scuff','scuffs','scum','scummy','second-class','second-tier','secretive','sedentary','seedy','seethe','seething','self-coup','self-criticism','self-defeating','self-destructive','self-humiliation','self-interest','self-interested','self-serving','selfinterested','selfish','selfishly','selfishness','semi-retarded','senile','sensationalize','senseless','senselessly','seriousness','sermonize','servitude','set-up','setback','setbacks','sever','severe','severity','sh*t','shabby','shadowy','shady','shake','shaky','shallow','sham','shambles','shame','shameful','shamefully','shamefulness','shameless','shamelessly','shamelessness','shark','sharply','shatter','shemale','shimmer','shimmy','shipwreck','shirk','shirker','shit','shiver','shock','shocked','shocking','shockingly','shoddy','short-lived','shortage','shortchange','shortcoming','shortcomings','shortness','shortsighted','shortsightedness','showdown','shrew','shriek','shrill','shrilly','shrivel','shroud','shrouded','shrug','shun','shunned','sick','sicken','sickening','sickeningly','sickly','sickness','sidetrack','sidetracked','siege','sillily','silly','simplistic','simplistically','sin','sinful','sinfully','sinister','sinisterly','sink','sinking','skeletons','skeptic','skeptical','skeptically','skepticism','sketchy','skimpy','skinny','skittish','skittishly','skulk','slack','slander','slanderer','slanderous','slanderously','slanders','slap','slashing','slaughter','slaughtered','slave','slaves','sleazy','slime','slog','slogged','slogging','slogs','sloooooooooooooow','sloooow','slooow','sloow','sloppily','sloppy','sloth','slothful','slow','slow-moving','slowed','slower','slowest','slowly','sloww','slowww','slowwww','slug','sluggish','slump','slumping','slumpping','slur','slut','sluts','sly','smack','smallish','smash','smear','smell','smelled','smelling','smells','smelly','smelt','smoke','smokescreen','smolder','smoldering','smother','smoulder','smouldering','smudge','smudged','smudges','smudging','smug','smugly','smut','smuttier','smuttiest','smutty','snag','snagged','snagging','snags','snappish','snappishly','snare','snarky','snarl','sneak','sneakily','sneaky','sneer','sneering','sneeringly','snob','snobbish','snobby','snobish','snobs','snub','so-cal','soapy','sob','sober','sobering','solemn','solicitude','somber','sore','sorely','soreness','sorrow','sorrowful','sorrowfully','sorry','sour','sourly','spade','spank','spendy','spew','spewed','spewing','spews','spilling','spinster','spiritless','spite','spiteful','spitefully','spitefulness','splatter','split','splitting','spoil','spoilage','spoilages','spoiled','spoilled','spoils','spook','spookier','spookiest','spookily','spooky','spoon-fed','spoon-feed','spoonfed','sporadic','spotty','spurious','spurn','sputter','squabble','squabbling','squander','squash','squeak','squeaks','squeaky','squeal','squealing','squeals','squirm','stab','stagnant','stagnate','stagnation','staid','stain','stains','stale','stalemate','stall','stalls','stammer','stampede','standstill','stark','starkly','startle','startling','startlingly','starvation','starve','static','steal','stealing','steals','steep','steeply','stench','stereotype','stereotypical','stereotypically','stern','stew','sticky','stiff','stiffness','stifle','stifling','stiflingly','stigma','stigmatize','sting','stinging','stingingly','stingy','stink','stinks','stodgy','stole','stolen','stooge','stooges','stormy','straggle','straggler','strain','strained','straining','strange','strangely','stranger','strangest','strangle','streaky','strenuous','stress','stresses','stressful','stressfully','stricken','strict','strictly','strident','stridently','strife','strike','stringent','stringently','struck','struggle','struggled','struggles','struggling','strut','stubborn','stubbornly','stubbornness','stuck','stuffy','stumble','stumbled','stumbles','stump','stumped','stumps','stun','stunt','stunted','stupid','stupidest','stupidity','stupidly','stupified','stupify','stupor','stutter','stuttered','stuttering','stutters','sty','stymied','sub-par','subdued','subjected','subjection','subjugate','subjugation','submissive','subordinate','subpoena','subpoenas','subservience','subservient','substandard','subtract','subversion','subversive','subversively','subvert','succumb','suck','sucked','sucker','sucks','sucky','sue','sued','sueing','sues','suffer','suffered','sufferer','sufferers','suffering','suffers','suffocate','sugar-coat','sugar-coated','sugarcoated','suicidal','suicide','sulk','sullen','sully','sunder','sunk','sunken','superficial','superficiality','superficially','superfluous','superstition','superstitious','suppress','suppression','surrender','susceptible','suspect','suspicion','suspicions','suspicious','suspiciously','swagger','swamped','sweaty','swelled','swelling','swindle','swipe','swollen','symptom','symptoms','syndrome','taboo','tacky','taint','tainted','tamper','tangle','tangled','tangles','tank','tanked','tanks','tantrum','tardy','tarnish','tarnished','tarnishes','tarnishing','tattered','taunt','taunting','tauntingly','taunts','taut','tawdry','taxing','tease','teasingly','tedious','tediously','temerity','temper','tempest','temptation','tenderness','tense','tension','tentative','tentatively','tenuous','tenuously','tepid','terrible','terribleness','terribly','terror','terror-genic','terrorism','terrorize','testily','testy','tetchily','tetchy','thankless','thicker','thirst','thorny','thoughtless','thoughtlessly','thoughtlessness','thrash','threat','threaten','threatening','threats','threesome','throb','throbbed','throbbing','throbs','throttle','thug','thumb-down','thumbs-down','thwart','time-consuming','timid','timidity','timidly','timidness','tin-y','tingled','tingling','tired','tiresome','tiring','tiringly','toil','toll','top-heavy','topple','torment','tormented','torrent','tortuous','torture','tortured','tortures','torturing','torturous','torturously','totalitarian','touchy','toughness','tout','touted','touts','toxic','traduce','tragedy','tragic','tragically','traitor','traitorous','traitorously','tramp','trample','transgress','transgression','trap','traped','trapped','trash','trashed','trashy','trauma','traumatic','traumatically','traumatize','traumatized','travesties','travesty','treacherous','treacherously','treachery','treason','treasonous','trick','tricked','trickery','tricky','trivial','trivialize','trouble','troubled','troublemaker','troubles','troublesome','troublesomely','troubling','troublingly','truant','tumble','tumbled','tumbles','tumultuous','turbulent','turmoil','twist','twisted','twists','two-faced','two-faces','tyrannical','tyrannically','tyranny','tyrant','ugh','uglier','ugliest','ugliness','ugly','ulterior','ultimatum','ultimatums','ultra-hardline','un-viewable','unable','unacceptable','unacceptablely','unacceptably','unaccessible','unaccustomed','unachievable','unaffordable','unappealing','unattractive','unauthentic','unavailable','unavoidably','unbearable','unbearablely','unbelievable','unbelievably','uncaring','uncertain','uncivil','uncivilized','unclean','unclear','uncollectible','uncomfortable','uncomfortably','uncomfy','uncompetitive','uncompromising','uncompromisingly','unconfirmed','unconstitutional','uncontrolled','unconvincing','unconvincingly','uncooperative','uncouth','uncreative','undecided','undefined','undependability','undependable','undercut','undercuts','undercutting','underdog','underestimate','underlings','undermine','undermined','undermines','undermining','underpaid','underpowered','undersized','undesirable','undetermined','undid','undignified','undissolved','undocumented','undone','undue','unease','uneasily','uneasiness','uneasy','uneconomical','unemployed','unequal','unethical','uneven','uneventful','unexpected','unexpectedly','unexplained','unfairly','unfaithful','unfaithfully','unfamiliar','unfavorable','unfeeling','unfinished','unfit','unforeseen','unforgiving','unfortunate','unfortunately','unfounded','unfriendly','unfulfilled','unfunded','ungovernable','ungrateful','unhappily','unhappiness','unhappy','unhealthy','unhelpful','unilateralism','unimaginable','unimaginably','unimportant','uninformed','uninsured','unintelligible','unintelligile','unipolar','unjust','unjustifiable','unjustifiably','unjustified','unjustly','unkind','unkindly','unknown','unlamentable','unlamentably','unlawful','unlawfully','unlawfulness','unleash','unlicensed','unlikely','unlucky','unmoved','unnatural','unnaturally','unnecessary','unneeded','unnerve','unnerved','unnerving','unnervingly','unnoticed','unobserved','unorthodox','unorthodoxy','unpleasant','unpleasantries','unpopular','unpredictable','unprepared','unproductive','unprofitable','unprove','unproved','unproven','unproves','unproving','unqualified','unravel','unraveled','unreachable','unreadable','unrealistic','unreasonable','unreasonably','unrelenting','unrelentingly','unreliability','unreliable','unresolved','unresponsive','unrest','unruly','unsafe','unsatisfactory','unsavory','unscrupulous','unscrupulously','unsecure','unseemly','unsettle','unsettled','unsettling','unsettlingly','unskilled','unsophisticated','unsound','unspeakable','unspeakablely','unspecified','unstable','unsteadily','unsteadiness','unsteady','unsuccessful','unsuccessfully','unsupported','unsupportive','unsure','unsuspecting','unsustainable','untenable','untested','unthinkable','unthinkably','untimely','untouched','untrue','untrustworthy','untruthful','unusable','unusably','unuseable','unuseably','unusual','unusually','unviewable','unwanted','unwarranted','unwatchable','unwelcome','unwell','unwieldy','unwilling','unwillingly','unwillingness','unwise','unwisely','unworkable','unworthy','unyielding','upbraid','upheaval','uprising','uproar','uproarious','uproariously','uproarous','uproarously','uproot','upset','upseting','upsets','upsetting','upsettingly','urgent','useless','usurp','usurper','utterly','vagrant','vague','vagueness','vain','vainly','vanity','vehement','vehemently','vengeance','vengeful','vengefully','vengefulness','venom','venomous','venomously','vent','vestiges','vex','vexation','vexing','vexingly','vibrate','vibrated','vibrates','vibrating','vibration','vice','vicious','viciously','viciousness','victimize','vile','vileness','vilify','villainous','villainously','villains','villian','villianous','villianously','villify','vindictive','vindictively','vindictiveness','violate','violation','violator','violators','violent','violently','viper','virulence','virulent','virulently','virus','vociferous','vociferously','volatile','volatility','vomit','vomited','vomiting','vomits','vulgar','vulnerable','wack','wail','wallow','wane','waning','wanton','war-like','warily','wariness','warlike','warned','warning','warp','warped','wary','washed-out','waste','wasted','wasteful','wastefulness','wasting','water-down','watered-down','wayward','weak','weaken','weakening','weaker','weakness','weaknesses','weariness','wearisome','weary','wedge','weed','weep','weird','weirdly','wheedle','whimper','whine','whining','whiny','whips','whore','whores','wicked','wickedly','wickedness','wild','wildly','wiles','wilt','wily','wimpy','wince','wobble','wobbled','wobbles','woe','woebegone','woeful','woefully','womanizer','womanizing','worn','worried','worriedly','worrier','worries','worrisome','worry','worrying','worryingly','worse','worsen','worsening','worst','worthless','worthlessly','worthlessness','wound','wounds','wrangle','wrath','wreak','wreaked','wreaks','wreck','wrest','wrestle','wretch','wretched','wretchedly','wretchedness','wrinkle','wrinkled','wrinkles','wrip','wripped','wripping','writhe','wrong','wrongful','wrongly','wrought','yawn','zap','zapped','zaps','zealot','zealous','zealously','zombie'])
# pos_words = set(['a+','abound','abounds','abundance','abundant','accessable','accessible','acclaim','acclaimed','acclamation','accolade','accolades','accommodative','accomodative','accomplish','accomplished','accomplishment','accomplishments','accurate','accurately','achievable','achievement','achievements','achievible','acumen','adaptable','adaptive','adequate','adjustable','admirable','admirably','admiration','admire','admirer','admiring','admiringly','adorable','adore','adored','adorer','adoring','adoringly','adroit','adroitly','adulate','adulation','adulatory','advanced','advantage','advantageous','advantageously','advantages','adventuresome','adventurous','advocate','advocated','advocates','affability','affable','affably','affectation','affection','affectionate','affinity','affirm','affirmation','affirmative','affluence','affluent','afford','affordable','affordably','afordable','agile','agilely','agility','agreeable','agreeableness','agreeably','all-around','alluring','alluringly','altruistic','altruistically','amaze','amazed','amazement','amazes','amazing','amazingly','ambitious','ambitiously','ameliorate','amenable','amenity','amiability','amiabily','amiable','amicability','amicable','amicably','amity','ample','amply','amuse','amusing','amusingly','angel','angelic','apotheosis','appeal','appealing','applaud','appreciable','appreciate','appreciated','appreciates','appreciative','appreciatively','appropriate','approval','approve','ardent','ardently','ardor','articulate','aspiration','aspirations','aspire','assurance','assurances','assure','assuredly','assuring','astonish','astonished','astonishing','astonishingly','astonishment','astound','astounded','astounding','astoundingly','astutely','attentive','attraction','attractive','attractively','attune','audible','audibly','auspicious','authentic','authoritative','autonomous','available','aver','avid','avidly','award','awarded','awards','awe','awed','awesome','awesomely','awesomeness','awestruck','awsome','backbone','balanced','bargain','beauteous','beautiful','beautifullly','beautifully','beautify','beauty','beckon','beckoned','beckoning','beckons','believable','believeable','beloved','benefactor','beneficent','beneficial','beneficially','beneficiary','benefit','benefits','benevolence','benevolent','benifits','best','best-known','best-performing','best-selling','better','better-known','better-than-expected','beutifully','blameless','bless','blessing','bliss','blissful','blissfully','blithe','blockbuster','bloom','blossom','bolster','bonny','bonus','bonuses','boom','booming','boost','boundless','bountiful','brainiest','brainy','brand-new','brave','bravery','bravo','breakthrough','breakthroughs','breathlessness','breathtaking','breathtakingly','breeze','bright','brighten','brighter','brightest','brilliance','brilliances','brilliant','brilliantly','brisk','brotherly','bullish','buoyant','cajole','calm','calming','calmness','capability','capable','capably','captivate','captivating','carefree','cashback','cashbacks','catchy','celebrate','celebrated','celebration','celebratory','champ','champion','charisma','charismatic','charitable','charm','charming','charmingly','chaste','cheaper','cheapest','cheer','cheerful','cheery','cherish','cherished','cherub','chic','chivalrous','chivalry','civility','civilize','clarity','classic','classy','clean','cleaner','cleanest','cleanliness','cleanly','clear','clear-cut','cleared','clearer','clearly','clears','clever','cleverly','cohere','coherence','coherent','cohesive','colorful','comely','comfort','comfortable','comfortably','comforting','comfy','commend','commendable','commendably','commitment','commodious','compact','compactly','compassion','compassionate','compatible','competitive','complement','complementary','complemented','complements','compliant','compliment','complimentary','comprehensive','conciliate','conciliatory','concise','confidence','confident','congenial','congratulate','congratulation','congratulations','congratulatory','conscientious','considerate','consistent','consistently','constructive','consummate','contentment','continuity','contrasty','contribution','convenience','convenient','conveniently','convience','convienient','convient','convincing','convincingly','cool','coolest','cooperative','cooperatively','cornerstone','correct','correctly','cost-effective','cost-saving','counter-attack','counter-attacks','courage','courageous','courageously','courageousness','courteous','courtly','covenant','cozy','creative','credence','credible','crisp','crisper','cure','cure-all','cushy','cute','cuteness','danke','danken','daring','daringly','darling','dashing','dauntless','dawn','dazzle','dazzled','dazzling','dead-cheap','dead-on','decency','decent','decisive','decisiveness','dedicated','defeat','defeated','defeating','defeats','defender','deference','deft','deginified','delectable','delicacy','delicate','delicious','delight','delighted','delightful','delightfully','delightfulness','dependable','dependably','deservedly','deserving','desirable','desiring','desirous','destiny','detachable','devout','dexterous','dexterously','dextrous','dignified','dignify','dignity','diligence','diligent','diligently','diplomatic','dirt-cheap','distinction','distinctive','distinguished','diversified','divine','divinely','dominate','dominated','dominates','dote','dotingly','doubtless','dreamland','dumbfounded','dumbfounding','dummy-proof','durable','dynamic','eager','eagerly','eagerness','earnest','earnestly','earnestness','ease','eased','eases','easier','easiest','easiness','easing','easy','easy-to-use','easygoing','ebullience','ebullient','ebulliently','ecenomical','economical','ecstasies','ecstasy','ecstatic','ecstatically','edify','educated','effective','effectively','effectiveness','effectual','efficacious','efficient','efficiently','effortless','effortlessly','effusion','effusive','effusively','effusiveness','elan','elate','elated','elatedly','elation','electrify','elegance','elegant','elegantly','elevate','elite','eloquence','eloquent','eloquently','embolden','eminence','eminent','empathize','empathy','empower','empowerment','enchant','enchanted','enchanting','enchantingly','encourage','encouragement','encouraging','encouragingly','endear','endearing','endorse','endorsed','endorsement','endorses','endorsing','energetic','energize','energy-efficient','energy-saving','engaging','engrossing','enhance','enhanced','enhancement','enhances','enjoy','enjoyable','enjoyably','enjoyed','enjoying','enjoyment','enjoys','enlighten','enlightenment','enliven','ennoble','enough','enrapt','enrapture','enraptured','enrich','enrichment','enterprising','entertain','entertaining','entertains','enthral','enthrall','enthralled','enthuse','enthusiasm','enthusiast','enthusiastic','enthusiastically','entice','enticed','enticing','enticingly','entranced','entrancing','entrust','enviable','enviably','envious','enviously','enviousness','envy','equitable','ergonomical','err-free','erudite','ethical','eulogize','euphoria','euphoric','euphorically','evaluative','evenly','eventful','everlasting','evocative','exalt','exaltation','exalted','exaltedly','exalting','exaltingly','examplar','examplary','excallent','exceed','exceeded','exceeding','exceedingly','exceeds','excel','exceled','excelent','excellant','excelled','excellence','excellency','excellent','excellently','excels','exceptional','exceptionally','excite','excited','excitedly','excitedness','excitement','excites','exciting','excitingly','exellent','exemplar','exemplary','exhilarate','exhilarating','exhilaratingly','exhilaration','exonerate','expansive','expeditiously','expertly','exquisite','exquisitely','extol','extoll','extraordinarily','extraordinary','exuberance','exuberant','exuberantly','exult','exultant','exultation','exultingly','eye-catch','eye-catching','eyecatch','eyecatching','fabulous','fabulously','facilitate','fair','fairly','fairness','faith','faithful','faithfully','faithfulness','fame','famed','famous','famously','fancier','fancinating','fancy','fanfare','fans','fantastic','fantastically','fascinate','fascinating','fascinatingly','fascination','fashionable','fashionably','fast','fast-growing','fast-paced','faster','fastest','fastest-growing','faultless','fav','fave','favor','favorable','favored','favorite','favorited','favour','fearless','fearlessly','feasible','feasibly','feat','feature-rich','fecilitous','feisty','felicitate','felicitous','felicity','fertile','fervent','fervently','fervid','fervidly','fervor','festive','fidelity','fiery','fine','fine-looking','finely','finer','finest','firmer','first-class','first-in-class','first-rate','flashy','flatter','flattering','flatteringly','flawless','flawlessly','flexibility','flexible','flourish','flourishing','fluent','flutter','fond','fondly','fondness','foolproof','foremost','foresight','formidable','fortitude','fortuitous','fortuitously','fortunate','fortunately','fortune','fragrant','free','freed','freedom','freedoms','fresh','fresher','freshest','friendliness','friendly','frolic','frugal','fruitful','ftw','fulfillment','fun','futurestic','futuristic','gaiety','gaily','gain','gained','gainful','gainfully','gaining','gains','gallant','gallantly','galore','geekier','geeky','gem','gems','generosity','generous','generously','genial','genius','gentle','gentlest','genuine','gifted','glad','gladden','gladly','gladness','glamorous','glee','gleeful','gleefully','glimmer','glimmering','glisten','glistening','glitter','glitz','glorify','glorious','gloriously','glory','glow','glowing','glowingly','god-given','god-send','godlike','godsend','gold','golden','good','goodly','goodness','goodwill','goood','gooood','gorgeous','gorgeously','grace','graceful','gracefully','gracious','graciously','graciousness','grand','grandeur','grateful','gratefully','gratification','gratified','gratifies','gratify','gratifying','gratifyingly','gratitude','great','greatest','greatness','grin','groundbreaking','guarantee','guidance','guiltless','gumption','gush','gusto','gutsy','hail','halcyon','hale','hallmark','hallmarks','hallowed','handier','handily','hands-down','handsome','handsomely','handy','happier','happily','happiness','happy','hard-working','hardier','hardy','harmless','harmonious','harmoniously','harmonize','harmony','headway','heal','healthful','healthy','hearten','heartening','heartfelt','heartily','heartwarming','heaven','heavenly','helped','helpful','helping','hero','heroic','heroically','heroine','heroize','heros','high-quality','high-spirited','hilarious','holy','homage','honest','honesty','honor','honorable','honored','honoring','hooray','hopeful','hospitable','hot','hotcake','hotcakes','hottest','hug','humane','humble','humility','humor','humorous','humorously','humour','humourous','ideal','idealize','ideally','idol','idolize','idolized','idyllic','illuminate','illuminati','illuminating','illumine','illustrious','ilu','imaculate','imaginative','immaculate','immaculately','immense','impartial','impartiality','impartially','impassioned','impeccable','impeccably','important','impress','impressed','impresses','impressive','impressively','impressiveness','improve','improved','improvement','improvements','improves','improving','incredible','incredibly','indebted','individualized','indulgence','indulgent','industrious','inestimable','inestimably','inexpensive','infallibility','infallible','infallibly','influential','ingenious','ingeniously','ingenuity','ingenuous','ingenuously','innocuous','innovation','innovative','inpressed','insightful','insightfully','inspiration','inspirational','inspire','inspiring','instantly','instructive','instrumental','integral','integrated','intelligence','intelligent','intelligible','interesting','interests','intimacy','intimate','intricate','intrigue','intriguing','intriguingly','intuitive','invaluable','invaluablely','inventive','invigorate','invigorating','invincibility','invincible','inviolable','inviolate','invulnerable','irreplaceable','irreproachable','irresistible','irresistibly','issue-free','jaw-droping','jaw-dropping','jollify','jolly','jovial','joy','joyful','joyfully','joyous','joyously','jubilant','jubilantly','jubilate','jubilation','jubiliant','judicious','justly','keen','keenly','keenness','kid-friendly','kindliness','kindly','kindness','knowledgeable','kudos','large-capacity','laud','laudable','laudably','lavish','lavishly','law-abiding','lawful','lawfully','lead','leading','leads','lean','led','legendary','leverage','levity','liberate','liberation','liberty','lifesaver','light-hearted','lighter','likable','like','liked','likes','liking','lionhearted','lively','logical','long-lasting','lovable','lovably','love','loved','loveliness','lovely','lover','loves','loving','low-cost','low-price','low-priced','low-risk','lower-priced','loyal','loyalty','lucid','lucidly','luck','luckier','luckiest','luckiness','lucky','lucrative','luminous','lush','luster','lustrous','luxuriant','luxuriate','luxurious','luxuriously','luxury','lyrical','magic','magical','magnanimous','magnanimously','magnificence','magnificent','magnificently','majestic','majesty','manageable','maneuverable','marvel','marveled','marvelled','marvellous','marvelous','marvelously','marvelousness','marvels','master','masterful','masterfully','masterpiece','masterpieces','masters','mastery','matchless','mature','maturely','maturity','meaningful','memorable','merciful','mercifully','mercy','merit','meritorious','merrily','merriment','merriness','merry','mesmerize','mesmerized','mesmerizes','mesmerizing','mesmerizingly','meticulous','meticulously','mightily','mighty','mind-blowing','miracle','miracles','miraculous','miraculously','miraculousness','modern','modest','modesty','momentous','monumental','monumentally','morality','motivated','multi-purpose','navigable','neat','neatest','neatly','nice','nicely','nicer','nicest','nifty','nimble','noble','nobly','noiseless','non-violence','non-violent','notably','noteworthy','nourish','nourishing','nourishment','novelty','nurturing','oasis','obsession','obsessions','obtainable','openly','openness','optimal','optimism','optimistic','opulent','orderly','originality','outdo','outdone','outperform','outperformed','outperforming','outperforms','outshine','outshone','outsmart','outstanding','outstandingly','outstrip','outwit','ovation','overjoyed','overtake','overtaken','overtakes','overtaking','overtook','overture','pain-free','painless','painlessly','palatial','pamper','pampered','pamperedly','pamperedness','pampers','panoramic','paradise','paramount','pardon','passion','passionate','passionately','patience','patient','patiently','patriot','patriotic','peace','peaceable','peaceful','peacefully','peacekeepers','peach','peerless','pep','pepped','pepping','peppy','peps','perfect','perfection','perfectly','permissible','perseverance','persevere','personages','personalized','phenomenal','phenomenally','picturesque','piety','pinnacle','playful','playfully','pleasant','pleasantly','pleased','pleases','pleasing','pleasingly','pleasurable','pleasurably','pleasure','plentiful','pluses','plush','plusses','poetic','poeticize','poignant','poise','poised','polished','polite','politeness','popular','portable','posh','positive','positively','positives','powerful','powerfully','praise','praiseworthy','praising','pre-eminent','precious','precise','precisely','preeminent','prefer','preferable','preferably','prefered','preferes','preferring','prefers','premier','prestige','prestigious','prettily','pretty','priceless','pride','principled','privilege','privileged','prize','proactive','problem-free','problem-solver','prodigious','prodigiously','prodigy','productive','productively','proficient','proficiently','profound','profoundly','profuse','profusion','progress','progressive','prolific','prominence','prominent','promise','promised','promises','promising','promoter','prompt','promptly','proper','properly','propitious','propitiously','pros','prosper','prosperity','prosperous','prospros','protect','protection','protective','proud','proven','proves','providence','proving','prowess','prudence','prudent','prudently','punctual','pure','purify','purposeful','quaint','qualified','qualify','quicker','quiet','quieter','radiance','radiant','rapid','rapport','rapt','rapture','raptureous','raptureously','rapturous','rapturously','rational','razor-sharp','reachable','readable','readily','ready','reaffirm','reaffirmation','realistic','realizable','reasonable','reasonably','reasoned','reassurance','reassure','receptive','reclaim','recomend','recommend','recommendation','recommendations','recommended','reconcile','reconciliation','record-setting','recover','recovery','rectification','rectify','rectifying','redeem','redeeming','redemption','refine','refined','refinement','reform','reformed','reforming','reforms','refresh','refreshed','refreshing','refund','refunded','regal','regally','regard','rejoice','rejoicing','rejoicingly','rejuvenate','rejuvenated','rejuvenating','relaxed','relent','reliable','reliably','relief','relish','remarkable','remarkably','remedy','remission','remunerate','renaissance','renewed','renown','renowned','replaceable','reputable','reputation','resilient','resolute','resound','resounding','resourceful','resourcefulness','respect','respectable','respectful','respectfully','respite','resplendent','responsibly','responsive','restful','restored','restructure','restructured','restructuring','retractable','revel','revelation','revere','reverence','reverent','reverently','revitalize','revival','revive','revives','revolutionary','revolutionize','revolutionized','revolutionizes','reward','rewarding','rewardingly','rich','richer','richly','richness','right','righten','righteous','righteously','righteousness','rightful','rightfully','rightly','rightness','risk-free','robust','rock-star','rock-stars','rockstar','rockstars','romantic','romantically','romanticize','roomier','roomy','rosy','safe','safely','sagacity','sagely','saint','saintliness','saintly','salutary','salute','sane','satisfactorily','satisfactory','satisfied','satisfies','satisfy','satisfying','satisified','saver','savings','savior','savvy','scenic','seamless','seasoned','secure','securely','selective','self-determination','self-respect','self-satisfaction','self-sufficiency','self-sufficient','sensation','sensational','sensationally','sensations','sensible','sensibly','sensitive','serene','serenity','sexy','sharp','sharper','sharpest','shimmering','shimmeringly','shine','shiny','significant','silent','simpler','simplest','simplified','simplifies','simplify','simplifying','sincere','sincerely','sincerity','skill','skilled','skillful','skillfully','slammin','sleek','slick','smart','smarter','smartest','smartly','smile','smiles','smiling','smilingly','smitten','smooth','smoother','smoothes','smoothest','smoothly','snappy','snazzy','sociable','soft','softer','solace','solicitous','solicitously','solid','solidarity','soothe','soothingly','sophisticated','soulful','soundly','soundness','spacious','sparkle','sparkling','spectacular','spectacularly','speedily','speedy','spellbind','spellbinding','spellbindingly','spellbound','spirited','spiritual','splendid','splendidly','splendor','spontaneous','sporty','spotless','sprightly','stability','stabilize','stable','stainless','standout','state-of-the-art','stately','statuesque','staunch','staunchly','staunchness','steadfast','steadfastly','steadfastness','steadiest','steadiness','steady','stellar','stellarly','stimulate','stimulates','stimulating','stimulative','stirringly','straighten','straightforward','streamlined','striking','strikingly','striving','strong','stronger','strongest','stunned','stunning','stunningly','stupendous','stupendously','sturdier','sturdy','stylish','stylishly','stylized','suave','suavely','sublime','subsidize','subsidized','subsidizes','subsidizing','substantive','succeed','succeeded','succeeding','succeeds','succes','success','successes','successful','successfully','suffice','sufficed','suffices','sufficient','sufficiently','suitable','sumptuous','sumptuously','sumptuousness','super','superb','superbly','superior','superiority','supple','support','supported','supporter','supporting','supportive','supports','supremacy','supreme','supremely','supurb','supurbly','surmount','surpass','surreal','survival','survivor','sustainability','sustainable','swank','swankier','swankiest','swanky','sweeping','sweet','sweeten','sweetheart','sweetly','sweetness','swift','swiftness','talent','talented','talents','tantalize','tantalizing','tantalizingly','tempt','tempting','temptingly','tenacious','tenaciously','tenacity','tender','tenderly','terrific','terrifically','thank','thankful','thinner','thoughtful','thoughtfully','thoughtfulness','thrift','thrifty','thrill','thrilled','thrilling','thrillingly','thrills','thrive','thriving','thumb-up','thumbs-up','tickle','tidy','time-honored','timely','tingle','titillate','titillating','titillatingly','togetherness','tolerable','toll-free','top','top-notch','top-quality','topnotch','tops','tough','tougher','toughest','traction','tranquil','tranquility','transparent','treasure','tremendously','trendy','triumph','triumphal','triumphant','triumphantly','trivially','trophy','trouble-free','trump','trumpet','trust','trusted','trusting','trustingly','trustworthiness','trustworthy','trusty','truthful','truthfully','truthfulness','twinkly','ultra-crisp','unabashed','unabashedly','unaffected','unassailable','unbeatable','unbiased','unbound','uncomplicated','unconditional','undamaged','undaunted','understandable','undisputable','undisputably','undisputed','unencumbered','unequivocal','unequivocally','unfazed','unfettered','unforgettable','unity','unlimited','unmatched','unparalleled','unquestionable','unquestionably','unreal','unrestricted','unrivaled','unselfish','unwavering','upbeat','upgradable','upgradeable','upgraded','upheld','uphold','uplift','uplifting','upliftingly','upliftment','upscale','usable','useable','useful','user-friendly','user-replaceable','valiant','valiantly','valor','valuable','variety','venerate','verifiable','veritable','versatile','versatility','vibrant','vibrantly','victorious','victory','viewable','vigilance','vigilant','virtue','virtuous','virtuously','visionary','vivacious','vivid','vouch','vouchsafe','warm','warmer','warmhearted','warmly','warmth','wealthy','welcome','well','well-backlit','well-balanced','well-behaved','well-being','well-bred','well-connected','well-educated','well-established','well-informed','well-intentioned','well-known','well-made','well-managed','well-mannered','well-positioned','well-received','well-regarded','well-rounded','well-run','well-wishers','wellbeing','whoa','wholeheartedly','wholesome','whooa','whoooa','wieldy','willing','willingly','willingness','win','windfall','winnable','winner','winners','winning','wins','wisdom','wise','wisely','witty','won','wonder','wonderful','wonderfully','wonderous','wonderously','wonders','wondrous','woo','work','workable','worked','works','world-famous','worth','worth-while','worthiness','worthwhile','worthy','wow','wowed','wowing','wows','yay','youthful','zeal','zenith','zest','zippy'])

def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    In this example, 'LOVE' and 'great' match the pos_words,
    and 'boring' matches the neg_words list.
    >>> feats = defaultdict(lambda: 0)
    >>> lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
    >>> sorted(feats.items())
    [('neg_words', 1), ('pos_words', 2)]
    """
    ###TODO
    feats['neg_words'] = 0
    feats['pos_words'] = 0
    for token in tokens:
    	if token.lower() in neg_words:
    		feats['neg_words'] += 1
    	if token.lower() in pos_words:
    		feats['pos_words'] += 1	
    pass


def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.

    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.

    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
    >>> feats
    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]
    """
    ###TODO
    feats = defaultdict()
    for function in feature_fns:
    	function(tokens, feats)
    return sorted(feats.items(), key = lambda x: x[0])
    pass


def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.

    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),

    When vocab is None, we build a new vocabulary from the given data.
    when vocab is not None, we do not build a new vocab, and we do not
    add any new terms to the vocabulary. This setting is to be used
    at test time.

    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]], dtype=int64)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
    """
    ###TODO
    
    #Counting features 
    featureCount = defaultdict(lambda: 0)
    feature = []
    featurelist = []
    for token in tokens_list:
    	feats = featurize(token, feature_fns)
    	feature.append(feats)
    	for feat, count in feats:
    		featureCount[feat] += 1

    #Building Vocabulary if vocab is None
    if vocab == None:
    	featurelist = []
    	for feat in featureCount:
    		if featureCount[feat] >= min_freq:
    			featurelist.append(feat)
    	sortedfeatures = sorted(featurelist)
    	vocab = {}
    	for count, feat in enumerate(sortedfeatures):
    		vocab[feat] = count

    #creating the sparse matrix
    row = []
    column = []
    data = []
    for i in range(len(feature)):
    	for j in range(len(feature[i])):
    		if feature[i][j][0] in vocab:
    			row.extend([i])
    			column.extend([vocab[feature[i][j][0]]])
    			data.extend([feature[i][j][1]])
    X = csr_matrix((data, (row, column)), shape=(len(tokens_list),len(vocab)), dtype=np.int64)
    return X, vocab
    pass


def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).

    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.

    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    ###TODO
    cross_validation = KFold(n_splits = k)
    accuracies = []
    for X_train_index, X_test_index in cross_validation.split(X):
    	clf.fit(X[X_train_index], labels[X_train_index])
    	predictions = clf.predict(X[X_test_index])
    	accuracies.append(accuracy_score(labels[X_test_index], predictions))
    return np.mean(accuracies)
    pass


def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.

    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.

    In addition to looping over possible assignments to
    keep_internal_punct and min_freqs, we will enumerate all
    possible combinations of feature functions. So, if
    feature_fns = [token_features, token_pair_features, lexicon_features],
    then we will consider all 7 combinations of features (see Log.txt
    for more examples).

    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])

    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.

      This list should be SORTED in descending order of accuracy.

      This function will take a bit longer to run (~20s for me).
    """
    ###TODO
    dictlist = []
    featureslist = []
    for i in range(len(feature_fns)):
    	for j in combinations(feature_fns, i + 1):
    		featureslist.append(list(j))
    for punct in punct_vals:
    	tokens_list = [tokenize(doc, punct) for doc in docs]
    	for frequency in min_freqs:
    		for features in featureslist:
    			X, vocab = vectorize(tokens_list, features, frequency)
    			accuracies = cross_validation_accuracy(LogisticRegression(solver='liblinear'), X, labels, 5)
    			dicts = {}
    			dicts['punct'] = punct
    			dicts['features'] = features
    			dicts['min_freq'] = frequency
    			dicts['accuracy'] = accuracies
    			dictlist.append((dicts))
    return sorted(dictlist, key=lambda x: (x['accuracy'],x['min_freq']), reverse = True)
    pass


def plot_sorted_accuracies(results):
    """
    Plot all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    Save to "accuracies.png".
    """
    ###TODO
    accuarcy = []
    for i in results:
    	accuarcy.append(i['accuracy'])
    sortedaccuracy = sorted(accuarcy)
    plt.xlabel('Setting')
    plt.ylabel('Accuracy')
    plt.plot(sortedaccuracy)
    plt.savefig('accuracies.png')
    pass


def mean_accuracy_per_setting(results):
    """
    To determine how important each model setting is to overall accuracy,
    we'll compute the mean accuracy of all combinations with a particular
    setting. For example, compute the mean accuracy of all runs with
    min_freq=2.

    Params:
      results...The output of eval_all_combinations
    Returns:
      A list of (accuracy, setting) tuples, SORTED in
      descending order of accuracy.
    """
    ###TODO
    meanlist = []
    count = defaultdict(list)
    for i in results:
    	feature_name = 'features=' + ' '.join(i.__name__ for i in i['features'])
    	punct_name = 'punct=' + str(i['punct'])
    	min_freq_name = 'min_freq=' + str(i['min_freq'])
    	count[feature_name].append(i['accuracy'])
    	count[punct_name].append(i['accuracy'])
    	count[min_freq_name].append(i['accuracy'])
    for setting, accuracy in count.items():
    	mean_accuracy = np.mean(accuracy)
    	meanlist.append((mean_accuracy, setting))
    return sorted(meanlist, key=lambda x: x[0], reverse = True)   
    pass


def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.
    (i.e., no cross-validation done here)

    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """
    ###TODO
    punct = best_result['punct']
    feature_fns = best_result['features']
    min_freq = best_result['min_freq']
    tokens_list = []
    for i in docs:
    	tokens_list.append(tokenize(i,punct))
    X, vocab = vectorize(tokens_list, feature_fns, min_freq, vocab=None)
    clf = LogisticRegression(solver='liblinear').fit(X, labels)
    return clf, vocab
    pass


def top_coefs(clf, label, n, vocab):
    """
    Find the n features with the highest coefficients in
    this classifier for this label.
    See the .coef_ attribute of LogisticRegression.

    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
    """
    ###TODO
    coeflist = []
    coef = clf.coef_[0]
    for feature in vocab:
    	coeflist.append((feature, coef[vocab[feature]]))
    
    if label == 1:
    	return sorted(coeflist, key=lambda x: -x[1])[:n]
       
    elif label == 0:
        sorted_coeflist = sorted(coeflist, key= lambda x: x[1])[:n]
        result_tuple = []
        for i in sorted_coeflist:
        	feature_name = i[0]
        	coefficient = -1 * i[1]
        	result_tuple.append((feature_name, coefficient))
        return result_tuple
    pass


def parse_test_data(best_result, vocab):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.

    Note: use read_data function defined above to read the
    test data.

    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    ###TODO
    punct = best_result['punct']
    feature_fns = best_result['features']
    min_freq = best_result['min_freq']
    test_docs, test_labels = read_data(os.path.join('data', 'test'))

    docs = [tokenize(doc, punct) for doc in test_docs]
    X_test, vocab=vectorize(docs, feature_fns, min_freq, vocab)
    return (test_docs, test_labels, X_test)
    pass


def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    """
    Print the n testing documents that are misclassified by the
    largest margin. By using the .predict_proba function of
    LogisticRegression <https://goo.gl/4WXbYA>, we can get the
    predicted probabilities of each class for each instance.
    We will first identify all incorrectly classified documents,
    then sort them in descending order of the predicted probability
    for the incorrect class.
    E.g., if document i is misclassified as positive, we will
    consider the probability of the positive class when sorting.

    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      clf...........LogisticRegression classifier fit on all training
                    data.
      n.............The number of documents to print.

    Returns:
      Nothing; see Log.txt for example printed output.
    """
    ###TODO
    proba = clf.predict_proba(X_test)
    predict = clf.predict(X_test)
    top_misclass = []
    for i in range(len(test_labels)):
    	if predict[i] != test_labels[i]:
    		if predict[i] == 0 :
    			top_misclass.append((test_labels[i], predict[i], proba[i][0], test_docs[i]))
    		else:
    			top_misclass.append((test_labels[i], predict[i], proba[i][1], test_docs[i]))
    sorted_topmisclass = sorted(top_misclass, key=lambda x: -x[2])
    for i in sorted_topmisclass[:n]:
    	print("\ntruth=%d predicted=%d proba=%f \n%s" % (i[0], i[1], i[2], i[3]))
    pass


def main():
    """
    Put it all together.
    ALREADY DONE.
    """
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)


if __name__ == '__main__':
    main()
