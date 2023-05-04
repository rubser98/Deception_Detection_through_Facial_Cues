#import matlab.engine
import numpy as np
import matplotlib.pyplot as mpl
import os
from os.path import isfile, join
import pandas as pd
import cv2 as cv
import struct

#restituisce indice facial landmark per ogni regione facciale di interesse per la simmetria

def region_of_interest() -> dict:
    #mapping roi con facial landmark openface
    lmk_map = {}
    
    lmk_map['eyebrow_sx'] = ['22','23','24','25','26']
    lmk_map['eyebrow_dx'] = ['17','18','19','20','21']
    lmk_map['lip_sx'] = ['52','53','54','55','56']
    lmk_map['lip_dx'] = ['50','49','48','59','58']
    lmk_map['face_dx'] = [str(i) for i in range(8) ]
    lmk_map['face_dx'].append('31')
    lmk_map['face_dx'].append('32')
    lmk_map['face_sx'] = [str(i) for i in range(9,17)]
    lmk_map['face_sx'].append('34')
    lmk_map['face_sx'].append('35')
    return lmk_map


#funzione opensource per leggere i file contenente l'HOG feature vector
#codice non scritto da me
def read_hog(filename, batch_size=5000):
    """
    Read HoG features file created by OpenFace.
    For each frame, OpenFace extracts 12 * 12 * 31 HoG features, i.e., num_features = 4464. These features are stored in row-major order.
    :param filename: path to .hog file created by OpenFace
    :param batch_size: how many rows to read at a time
    :return: is_valid, hog_features
        is_valid: ndarray of shape [num_frames]
        hog_features: ndarray of shape [num_frames, num_features]
    """
    all_feature_vectors = []
    with open(filename, "rb") as f:
        num_cols, = struct.unpack("i", f.read(4))
        num_rows, = struct.unpack("i", f.read(4))
        num_channels, = struct.unpack("i", f.read(4))

        # The first four bytes encode a boolean value whether the frame is valid
        num_features = 1 + num_rows * num_cols * num_channels
        feature_vector = struct.unpack("{}f".format(num_features), f.read(num_features * 4))
        feature_vector = np.array(feature_vector).reshape((1, num_features))
        all_feature_vectors.append(feature_vector)

        # Every frame contains a header of four float values: num_cols, num_rows, num_channels, is_valid
        num_floats_per_feature_vector = 4 + num_rows * num_cols * num_channels
        # Read in batches of given batch_size
        num_floats_to_read = num_floats_per_feature_vector * batch_size
        # Multiply by 4 because of float32
        num_bytes_to_read = num_floats_to_read * 4

        while True:
            bytes = f.read(num_bytes_to_read)
            # For comparison how many bytes were actually read
            num_bytes_read = len(bytes)
            assert num_bytes_read % 4 == 0, "Number of bytes read does not match with float size"
            num_floats_read = num_bytes_read // 4
            assert num_floats_read % num_floats_per_feature_vector == 0, "Number of bytes read does not match with feature vector size"
            num_feature_vectors_read = num_floats_read // num_floats_per_feature_vector

            feature_vectors = struct.unpack("{}f".format(num_floats_read), bytes)
            # Convert to array
            feature_vectors = np.array(feature_vectors).reshape((num_feature_vectors_read, num_floats_per_feature_vector))
            # Discard the first three values in each row (num_cols, num_rows, num_channels)
            feature_vectors = feature_vectors[:, 3:]
            # Append to list of all feature vectors that have been read so far
            all_feature_vectors.append(feature_vectors)

            if num_bytes_read < num_bytes_to_read:
                break

        # Concatenate batches
        all_feature_vectors = np.concatenate(all_feature_vectors, axis=0)

        # Split into is-valid and feature vectors
        is_valid = all_feature_vectors[:, 0]
        feature_vectors = all_feature_vectors[:, 1:]

        return is_valid, feature_vectors

#funzione che richiama la libreria openface nel calcolo dei facial landmark e mascheramento della faccia nell'immagine
def align():
    path='C:\\Users\\Administrator\\OneDrive - Sogei\\Desktop\\Tesi\\Frame\\Truthful'
    cmd = 'cmd /c "FeatureExtraction.exe -fdir "'
    for dir in os.listdir(path):
        os.system(cmd+path+'\\'+dir+'" -out_dir "'+path+'\\'+dir+'""')
        print(dir)
    
#dato un HoG file restituisce l'HoG sotto formato di array numpy
def hog_extraction(hog_file: str) -> np.ndarray:
    #eng = matlab.engine.start_matlab()
    #hog_data = eng.Read_HOG_file(hog_file)
    _, hog_data = read_hog(hog_file) 
    hog = np.array(hog_data)
    hog_array = np.array(hog[0])
    return hog_array

#concatenazione hog e asimmetria in file csv 
def feature_fusion(truth: bool):
    f = 'C:\\Users\\Administrator\\OneDrive - Sogei\\Desktop\\Tesi\\Frame\\'
    path_out = 'C:\\Users\\Administrator\\OneDrive - Sogei\\Desktop\\Tesi\\Dataset\\Trial_finale\\'
    if truth:
        f = f+'Truthful\\'
        path_out = path_out+'Truthful\\'
    else:
        f = f+'Deceptive\\'
        path_out = path_out+'Deceptive\\'
    i=1
    lista_file_no_simmetria = ['trial_lie_041','trial_lie_043','trial_lie_045','trial_lie_053','trial_lie_057','trial_lie_060','trial_truth_016','trial_truth_017','trial_truth_029','trial_truth_041','trial_truth_042']
    for dir in os.listdir(f):
        print(dir)
        if dir not in lista_file_no_simmetria and i > 0:
            path_hog = f + dir +'\\'+ dir +'_aligned\\'
            f_symmetry = f + dir +'\\'+dir+'_symmetry.csv'
            df_symmetry = pd.read_csv(f_symmetry,sep=';',header=0)
            n_frames= len(df_symmetry.index)
            feature_df = pd.DataFrame()
            for i in range(n_frames):
                str_i = str(i+1)
                if i+1 < 10:
                    str_i = '000'+str_i
                elif i+1 < 100:
                    str_i = '00'+str_i
                elif i+1 < 1000:
                    str_i = '0'+str_i
                hog_file = path_hog+'frame_det_00_00'+str_i+'.hog'
                print(hog_file)
                feature_vector = hog_extraction(hog_file)
                eyebrow_sym = df_symmetry['symmetry_eyebrows'][i]
                lips_sym = df_symmetry['symmetry_lips'][i]
                #feature_vector = np.append(feature_vector, eyebrow_sym)
                #feature_vector= np.append(feature_vector, lips_sym)
                print(feature_vector)
                feature = {i : feature_vector[i] for i in range(len(feature_vector))}
                feature[4464] = eyebrow_sym
                feature[4465] = lips_sym
                feature_df = feature_df.append(feature,ignore_index=True)
                
            feature_df.to_csv(path_out+dir+'.csv',sep=';')
        i+=1
            

 #feature_fusion(True)


    

#hf = 'C:\\Users\\Administrator\\OneDrive - Sogei\\Desktop\\Tesi\\Frame\\Deceptive\\trial_lie_001\\trial_lie_001_aligned\\'
#hog_extraction(hf+'frame_det_00_000004.hog')
#align()


#funzione che richiama la libreria OpenFace per estrarre le hog feature da ogni frame
'''
def hog_per_frame():
    path='C:\\Users\\Administrator\\OneDrive - Sogei\\Desktop\\Tesi\\Frame\\Deceptive'
    
    
    #os.system('cmd /k "cd C:/Users/Administrator/"OneDrive - Sogei"/Desktop/Tesi/OpenFace_2.2.0_win_x64"')
    #os.system('cmd /k "dir"')
    cmd = 'cmd /c "FeatureExtraction.exe -f "'
    i=0
    for dir in os.listdir(path):
        
        if i == 30:
            print(dir)
            for file in os.listdir(path+'\\'+dir+'\\'+dir+'_aligned'):
                
                if file.endswith('.bmp'):
                    
                    #print(path+'\\'+dir+'\\'+dir+'_aligned\\'+file)
                    os.system(cmd+path+'\\'+dir+'\\'+dir+'_aligned\\'+file+'" -hogalign -out_dir "'+path+'\\'+dir+'\\'+dir+'_aligned""')
                    #print(cmd+path+'\\'+dir+'_aligned\\'+file+'" -hogalign -out_dir "'+path+'\\'+dir+'_aligned""')
        i+=1
  
#hog_per_frame()
'''

#funzione che restituisce il numero di pixel dato un insieme di punti e le coordinate minime e massime
def pixel_area(points):
    xmin = xmax = points[0][0]
    ymin = ymax = points[0][1]
    
    for p in points:
        if p[0] > xmax:
            xmax = p[0]
        if p[0] < xmin:
            xmin = p[0]
        if p[1] > ymax:
            ymax = p[1]
        if p[1] < ymin:
            ymin = p[1]
    
    base = xmax - xmin
    altezza = ymax - ymin
    
    return xmin,base,ymin,altezza, base * altezza

#funzione che restituisce l'insieme dei punti di una regione facciale di uno specifico frame di un video
def lndk_point(ind,df,region) -> list:
    pnt=[]
    roi = region_of_interest()
    for i in range(len(roi[region])):
        x = round(df[' x_'+roi[region][i]][ind])
        y = round(df[' y_'+roi[region][i]][ind])
        pnt.append((x,y))
    return pnt

#la funzione restituisce il flow magnitude di una roi tra frame t-1 e t
def optical_flow(prev,frm,ind_prev,ind_frm,region,land):
    #leggo il frame t
    img_frm = cv.imread(frm)
    #leggo il frame t-1
    img_prev = cv.imread(prev)
    #lista di punti dei facial landmarks di una roi del frame t-1
    pnt_prev= lndk_point(ind_prev,land,region)
    xmin_prev,b_prev,ymin_prev,h_prev, area_prev = pixel_area(pnt_prev)
    #ritaglio il frame t-1 prendendo in considerazione solo la roi
    #lista di punti dei facial landmarks di una roi del frame t
    pnt_frm= lndk_point(ind_frm,land,region)
    xmin_frm,b_frm,ymin_frm,h_frm, area_frm = pixel_area(pnt_frm)
    #settaggio valori per avere immagini delle stesse dimensioni da dare all'optical flow
    #setto base e altezza con il massimo per avere ritagliato
    b = max(b_prev,b_frm)
    h = max(h_prev,h_frm)
    #setto xmin e ymin 
    xmin = min(xmin_frm,xmin_prev)
    ymin = min(ymin_frm,ymin_prev)
    #ritaglio il frame t prendendo in considerazione solo la roi
    img_frm_region = img_frm[ymin:ymin+h, xmin: xmin+b].copy()
    img_prev_region = img_prev[ymin:ymin+h, xmin: xmin+b].copy()
    #frame t-1 e t ritagliati da rgb a gray
    gray_frm = cv.cvtColor(img_frm_region, cv.COLOR_BGR2GRAY)
    gray_prev = cv.cvtColor(img_prev_region, cv.COLOR_BGR2GRAY)
    #frame t ritagliato da rgb a hsv, set saturation
    hsv = np.zeros_like(img_prev_region)
    hsv[..., 1] = 255
    hsv[:,:,1] = cv.cvtColor(img_prev_region, cv.COLOR_RGB2HSV)[:,:,1]
    #obtain dense optical flow paramters
    flow = cv.calcOpticalFlowFarneback(gray_prev, gray_frm, flow=None,
                                        pyr_scale=0.5, levels=1, winsize=15,
                                        iterations=2,
                                        poly_n=5, poly_sigma=1.1, flags=0)
    #convert from cartesian to polar
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    #hue corresponds to direction
    hsv[:,:,0] = ang * (180/ np.pi / 2)
    #value corresponds to magnitude
    hsv[:,:,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    #convert HSV to int32's
    hsv = np.asarray(hsv, dtype= np.float32)
    rgb_flow = cv.cvtColor(hsv,cv.COLOR_HSV2RGB)
    #cv.imshow('rgbflow',rgb_flow)
    #cv.waitKey(0)
    return np.mean(mag)

#prev='C:\\Users\\Administrator\\OneDrive - Sogei\\Desktop\\Tesi\\Frame\\Deceptive\\trial_lie_001\\image0.jpg'
#frm='C:\\Users\\Administrator\\OneDrive - Sogei\\Desktop\\Tesi\\Frame\\Deceptive\\trial_lie_001\\image1.jpg'
#ffile='C:\\Users\\Administrator\\OneDrive - Sogei\\Desktop\\Tesi\\Frame\\Deceptive\\trial_lie_001\\trial_lie_001.csv'
#print(optical_flow(prev,frm,0,1,'lip_sx',ffile))




        
#restuisce array con i flow magnitude di ogni regione
# magnitude[0] = flow magnitude labbro sinistro
# magnitude[1] = flow magnitude labbro destro
# magnitude[2] = flow magnitude sopracciglio sinistro
# magnitude[3] = flow magnitude sopracciglio destro
def optical_flow_magnitude(prev, img,ind_prev,ind_img,file_landmark):
    land = pd.read_csv(file_landmark,sep=',',header=0)
    magnitude = []
    magnitude.append(optical_flow(prev,img,ind_prev,ind_img,'lip_sx',land))
    magnitude.append(optical_flow(prev,img,ind_prev,ind_img,'lip_dx',land))
    magnitude.append(optical_flow(prev,img,ind_prev,ind_img,'eyebrow_sx',land))
    magnitude.append(optical_flow(prev,img,ind_prev,ind_img,'eyebrow_dx',land))
    return magnitude

#calcola movement score di una roi
def movement_score_roi(magnitude,df,i,region):
    points = lndk_point(i,df,region)

    _, _, _, _, M = pixel_area(points)
    return magnitude/M

#calcola movement score di ogni roi
#m_scores[0] = movement score labbro sinistro
#m_scores[1] = movement score labbro destro
#m_scores[2] = movement score sopracciglio sinistro
#m_scores[3] = movement score sopracciglio destro
def movement_score(magnitude,f,i):
    df = pd.read_csv(f, sep=',', header=0)
    m_scores = []
    m_scores.append(movement_score_roi(magnitude[0],df,i,'lip_sx'))
    m_scores.append(movement_score_roi(magnitude[1],df,i,'lip_dx'))
    m_scores.append(movement_score_roi(magnitude[2],df,i,'eyebrow_sx'))
    m_scores.append(movement_score_roi(magnitude[3],df,i,'eyebrow_dx'))
    return m_scores

#calcola il valore di simmetria delle due regioni facciali in analisi 
def symmetry_score(mscores):
    lamda = 3.8
    symmetry_score_eyebrows = 1 - lamda * abs(mscores[0]-mscores[1])
    symmetry_score_lips = 1 - lamda * abs(mscores[2]-mscores[3])
    #verifico limite superiore symmetry score
    if symmetry_score_eyebrows > 1:
        symmetry_score_eyebrows = 1
    if symmetry_score_lips > 1:
        symmetry_score_lips = 1
    #verifico limite inferiore symmetry score
    if symmetry_score_eyebrows < 0:
        symmetry_score_eyebrows = 0
    if symmetry_score_lips < 0:
        symmetry_score_lips = 0
    return symmetry_score_eyebrows, symmetry_score_lips

'''
magn = optical_flow_magnitude(prev,frm,0,1,ffile)
m_scores = movement_score(magn,ffile,1)
print(symmetry_score(m_scores))
'''
#calcola symmetry score per ogni frame di un video e lo salva in un csv
def symmetry_score_video(path: str,vid) -> None:
    df = pd.DataFrame(columns = ['frame','symmetry_eyebrows','symmetry_lips'])
    record = {'frame': 1,'symmetry_eyebrows': 0,'symmetry_lips': 0}
    images = [f for f in os.listdir(path+vid) if f.endswith('.jpg')]
    csv = path+vid+'\\'+[f for f in os.listdir(path+vid) if f.endswith('.csv')][0]
    n_frames= len(images)
    df = df.append(record,ignore_index=True)
    i=2
    prev = path+vid+'\\image0.jpg'
    ind_prev = 0
    while i < n_frames+1:
        ind_frm = ind_prev + 1
        frm = path+vid+'\\image'+str(i-1)+'.jpg'
        magn = optical_flow_magnitude(prev,frm,ind_prev,ind_frm,csv)
        m_scores = movement_score(magn,csv,i-1)
        se,sl = symmetry_score(m_scores)
        record = record = {'frame': i,'symmetry_eyebrows': se,'symmetry_lips': sl}
        df = df.append(record,ignore_index=True)
        prev = frm
        ind_prev = ind_frm
        i+=1
    df.to_csv(path+vid+'\\'+vid+'_symmetry_v2.csv',sep=';', index = False)

#vid='C:\\Users\\Administrator\\OneDrive - Sogei\\Desktop\\Tesi\\Frame\\Truthful\\'
#calcola symmetry score per ogni video di una cartella
def symmetry_score_all(path: str) -> None:
    for dir in os.listdir(path):
        print(dir)
        symmetry_score_video(path,dir)

#symmetry_score_all(vid)