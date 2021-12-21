import numpy as np

# Few notes... MJ111 and MJ116 are both actually trained with MBP_094. 
# Left is actually right hand..
# Right is actually left hand..
mime_left_labels = {}
mime_left_labels['2'] = 'Reach'
mime_left_labels['4'] = 'Grasp and Lift'
mime_left_labels['5'] = 'Reach'
mime_left_labels['6'] = 'Reach'
mime_left_labels['7'] = 'Reach'
mime_left_labels['8'] = 'Return'
mime_left_labels['9'] = 'Place'
mime_left_labels['10'] = 'Reach'
mime_left_labels['11'] = 'Place and Return'
mime_left_labels['12'] = 'Reach'
mime_left_labels['13'] = 'Reach'
mime_left_labels['14'] = 'Reach'
mime_left_labels['15'] = 'Push'
mime_left_labels['16'] = 'Return'
mime_left_labels['17'] = 'Place and Return'
mime_left_labels['18'] = 'Reach'
mime_left_labels['19'] = 'Reach'
mime_left_labels['20'] = 'Reach'
mime_left_labels['21'] = 'Push'
mime_left_labels['22'] = 'Reach'
mime_left_labels['23'] = 'Reach'
mime_left_labels['24'] = 'Place and Return'
mime_left_labels['25'] = 'Push'
mime_left_labels['26'] = 'Return'
mime_left_labels['27'] = 'Reach'
mime_left_labels['28'] = 'Reach'
mime_left_labels['29'] = 'Reach'
mime_left_labels['30'] = 'Push'
mime_left_labels['31'] = 'Reach' 
mime_left_labels['32'] = 'Grasp and Lift'
mime_left_labels['33'] = 'Grasp and Lift'
mime_left_labels['35'] = 'Grasp and Lift'
mime_left_labels['36'] = 'Place'
mime_left_labels['37'] = 'Push'
mime_left_labels['38'] = 'Place and Return'
mime_left_labels['39'] = 'Push'
mime_left_labels['41'] = 'Push'
mime_left_labels['42'] = 'Place and Return'
mime_left_labels['43'] = 'Place and Return'
mime_left_labels['44'] = 'Place and Return'
mime_left_labels['46'] = 'Push'
mime_left_labels['47'] = 'Place and Return'
mime_left_labels['48'] = 'Place and Return'
mime_left_labels['49'] = 'Place and Return'
mime_left_labels['50'] = 'Place and Return'
mime_left_labels['53'] = 'Place and Return'
mime_left_labels['55'] = 'Push'
mime_left_labels['56'] = 'Push'
mime_left_labels['57'] = 'Place and Return'
mime_left_labels['59'] = 'Push'
mime_left_labels['60'] = 'Push'
mime_left_labels['66'] = 'Grasp and Lift'
mime_left_labels['70'] = 'Push'

mime_left_inverse_labels = {}
for k,v in mime_left_labels.items():
    print(k,v)
    if v not in mime_left_inverse_labels.keys():
        mime_left_inverse_labels[v] = [k]
    else:
        mime_left_inverse_labels[v].append(k)
