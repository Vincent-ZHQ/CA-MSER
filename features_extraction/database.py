import os
from collections import defaultdict, OrderedDict

"""
The keys of the following dictionaries refer to specific emotion codenames
    used in each database.
Each entry is a list of aliases which can be used to refer to 
    the corresponding emotion during database initialization.
"""
IEMOCAP_EMO_CODES = {'neu': ['neu', 'neutral'],
                     'hap': ['hap', 'happy', 'happiness'],
                     'sad': ['sad', 'sadness'],
                     'ang': ['ang', 'angry', 'anger'],
                     'sur': ['sur', 'surprise', 'surprised'],
                     'fea': ['fea', 'fear'],
                     'dis': ['dis', 'disgust', 'disgusted'],
                     'fru': ['fru', 'frustrated', 'frustration'],
                     'exc': ['exc', 'excited', 'excitement'],
                     'oth': ['oth', 'other', 'others']}

EMODB_EMO_CODES =   {'N': ['neu', 'neutral'],
                     'F': ['hap', 'happy', 'happiness'],
                     'T': ['sad', 'sadness'],
                     'W': ['angry','anger'],
                     'L': ['bored', 'boredom'],
                     'A': ['fea', 'fear'],
                     'E': ['dis', 'disgust', 'disgusted']}


class IEMOCAP_database():
    """
    IEMOCAP database contains data from 10 actors, 5 male and 5 female,
    during their affective dyadic interaction. The database consists of
    5 sessions, containing both improvised and scripted sessions. Each session
    consists of 2 unique speakers: 1 male and 1 female.

    For each session, the utterances are organized into conversation folders
        eg. Ses01F_impro01/                     -> improvised conversation 01 of Session 01
                |-- Ses01F_impro01_F000.wav     -> speaker F, utterance 000
                |-- Ses01F_impro01_M000.wav     -> speaker M, utterance 000
                |-- ...

    This function extract utterance filenames and labels for improvised sessions,
    organized into dictionary of {'speakerID':[(conversation_wavs,lab),(wavs,lab),...,(wavs,lab)]}

        > speakerID eg. 1M: Session 1, Male speaker
    
    Database Reference:
        (2008). IEMOCAP: Interactive emotional dyadic motion capture database. 
        Language Resources and Evaluation.
    
    Authors:
        Busso, Carlos
        Bulut, Murtaza
        Lee, Chi-Chun
        Kazemzadeh, Abe
        Mower, Emily
        Kim, Samuel
        Chang, Jeannette
        Lee, Sungbok
        Narayanan, Shrikanth
    
    Download request link:
        https://sail.usc.edu/iemocap/iemocap_release.htm
    """

    def __init__(self, database_dir, emot_map = {'ang': 0, 'sad':1, 'hap':2, 'neu':3},
                        include_scripted=False): 
        
        #Path
        self.database_dir = database_dir

        #Emotion to label mapping for features
        self.emot_map = emot_map

        #IEMOCAP Session name
        self.sessions = ['Session1','Session2','Session3','Session4','Session5']
        # self.sessions = ['Session3','Session4','Session5']

        #IEMOCAP available emotion classes 
        self.all_emo_classes = IEMOCAP_EMO_CODES.keys()

        #to include scripted session
        self.include_scripted = include_scripted

    def get_speaker_id(self, session, gender):

        return session[-1]+gender
    
    def get_classes(self):

        classes={}
        for key,value in self.emot_map.items():
            if value in classes.keys():
                classes[value] += '+'+key
            else:
                classes[value] = key
        
        return classes

    def get_files(self):
        """
        Get all the required .wav file paths for each speaker and organized into
            dictionary:
                keys   -> speaker ID
                values -> list of (.wav filepath, label) tuples for corresponding speaker
        """
        emotions = self.emot_map.keys()
        dataset_dir = self.database_dir
        all_speaker_files = defaultdict()
        total_num_files = 0
        for session_name in os.listdir(dataset_dir):
           
            if session_name not in self.sessions:
                continue
            wav_dir = os.path.join(dataset_dir, session_name, "sentences/wav")
            lab_dir = os.path.join(dataset_dir, session_name, "dialog/EmoEvaluation")

            M_wav, F_wav = list(), list()
            for conversation_folder in os.listdir(wav_dir):
                #omit hidden folders
                if conversation_folder.startswith('.'):
                    continue

                if self.include_scripted == False:
                    # Only use improvised data, for example ".../wav/Ses01F_impro01"
                    if conversation_folder[7:12] != "impro":
                        continue
                
                # Path to the directory containing all the *.wav files of the
                # current conversation
                conversation_dir = os.path.join(wav_dir, conversation_folder)
                
                # Get labels of all utterance in the current conversation
                label_path = os.path.join(lab_dir, conversation_folder + ".txt")
                labels = dict() 
                with open(label_path, "r") as fin:
                    #print(label_path)
                    for line in fin:
                        # If this line is sth like
                        # [6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2.5000, 2.5000, 2.5000]
                        if line[0] == "[":
                            t = line.split()
                            # For e.g., {"Ses01F_impro01_F000": "neu", ...}
                            labels[t[3]] = t[4]
                
                # Get a list of paths to all *.wav files
                wav_files = []
                for wav_name in os.listdir(conversation_dir):
                    #omit hidden folders
                    if wav_name.startswith('.'):
                        continue
                    #omit non .wav files
                    name, ext = os.path.splitext(wav_name)
                    if ext != ".wav":
                        continue
                    #emotion label
                    emotion = labels[name]
                    if emotion not in emotions:
                        continue
                    label = self.emot_map[emotion]
                    
                    wav_files.append((os.path.join(conversation_dir, wav_name), label))
                
                #separate into individual speakers
                F_wav.extend([emowav for emowav in wav_files if emowav[0][-8] == "F"])
                #F_lab.append(label_path)
                M_wav.extend([emowav for emowav in wav_files if emowav[0][-8] == "M"])
                #M_lab.append(label_path)
                
            #Put speaker utterance and label paths into dictionary
            all_speaker_files[self.get_speaker_id(session_name,'M')] = M_wav
            all_speaker_files[self.get_speaker_id(session_name,'F')] = F_wav

            total_num_files += len(M_wav) + len(F_wav)
        print(f'\nNUMBER OF FILES: {total_num_files}\n')
        return all_speaker_files


class EMODB_database():
    """
    emoDB database contains emotional speech acted by 10 actors (5 male, 5 female).
    Each actor simulated 7 emotions, producing 10 utterances per emotion (5 short, 5 longer
    utterances).

    Database Reference:
        (2005). A database of German emotional speech. 9th European Conference on 
        Speech Communication and Technology. 5. 1517-1520.

    Authors:
        Burkhardt, Felix
        Paeschke, Astrid
        Rolfes, M.
        Sendlmeier, Walter
        Weiss, Benjamin 
    """
    
    def __init__(self, database_dir, emot_map={'neutral':0,'happy':1,'sad':2,'angry':3,'bored':4,'fear':5,'disgust':6}):
        
        #Path
        self.database_dir = database_dir

        #Emotion to label mapping
        self.emot_map = emot_map

        self.all_emo_codes = EMODB_EMO_CODES

    
    def _get_speaker_id(self,speaker_code):
        emodb_speaker_ids = {'03':'1M','08':'1F','09':'2F','10':'2M','11':'3M',
                             '12':'4M','13':'3F','14':'4F','15':'5M','16':'5F'}

        return emodb_speaker_ids[speaker_code] 
    
    def get_classes(self):

        classes={}
        for key,value in self.emot_map.items():
            if value in classes.keys():
                classes[value] += '+'+key
            else:
                classes[value] = key
        
        return classes


    def get_files(self):
        
        all_files =defaultdict(list)

        #Get all filenames
        filenames = []
        for (dirpath, dirnames, fnames) in os.walk(self.database_dir):
            for name in fnames:
                if name.startswith('.'):
                    continue
                filenames.append(name)
           
        assert len(filenames) == 535

        emotion_codes_to_labels = {}
        #Map the required emotion codes to class label
        for clss in self.emot_map.keys():
            for emo_code, emo_class in self.all_emo_codes.items():
                if clss in emo_class:
                    emotion_codes_to_labels[emo_code] = self.emot_map[clss]
        
        #Grouped the files based on speaker, into dictionary
        for fname in filenames:
            speaker_code = fname[0:2]
            emotion_code = fname[5]

            speaker_id = self._get_speaker_id(speaker_code)
            emotion_label = emotion_codes_to_labels[emotion_code]

            all_files[speaker_id].append((dirpath+fname,emotion_label))
        
        return all_files
    '''
    #remove hidden files
    filenames = [f for f in filenames if not f[0] == '.']
    print(f'Total of {len(filenames)} files in directory {data_dir}')

    #Put filename into ID and labels
    all_filenames = []
    all_labels = []
    for files in filenames:
        speaker = files[0:2]
        emotion = files[5]
        file_id = files.replace('.wav','')

        if speaker[0] != '.' and speaker in emodb_speaker_ids:
            emotion_idx = emodb_emo_to_idx[emotion]
            all_filenames.append(file_id)
            all_labels.append(emotion_idx)

    assert len(all_filenames) == len(filenames)
    assert len(all_labels) ==  len(filenames)   

    return all_filenames, all_labels    
    '''
SER_DATABASES = {'IEMOCAP': IEMOCAP_database,
                 'EMODB'  : EMODB_database}
    
"""
### TESTING ###

database = IEMOCAP_database('/Volumes/AIWorks/IEMOCAP/IEMOCAP_full_release/')
all_speakers = database.get_files()
tot = 0
for sp in all_speakers.keys():
    tot += len(all_speakers[sp])
print(tot)

from features_extraction.features_util import extract_features
params={'window'        : 'hamming',
            'win_length'    : 40,
            'hop_length'    : 10,
            'ndft'          : 800,
            'nfreq'         : 200,
            'nmel'          : 128,
            'segment_size'  : 300
            }
features = extract_features(all_speakers,'logspec',params)
"""
    
    