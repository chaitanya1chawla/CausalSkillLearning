import numpy as np

# Few notes... MJ111 and MJ116 are both actually trained with MBP_094. 
# Left is actually right hand..
# Right is actually left hand..
mime_right_labels = {}
mime_right_labels['0'] = 'Grasp and Lift'
mime_right_labels['1'] = 'Reach'
mime_right_labels['2'] = 'Reach'
mime_right_labels['3'] = 'Reach'
mime_right_labels['4'] = 'Grasp and Lift'
mime_right_labels['7'] = 'Return'
mime_right_labels['8'] = 'Reach'
mime_right_labels['9'] = 'Grasp and Lift'
mime_right_labels['10'] = 'Return'
mime_right_labels['11'] = 'Reach'
mime_right_labels['12'] = 'Reach'
mime_right_labels['13'] = 'Reach'
mime_right_labels['14'] = 'Grasp and Lift'
mime_right_labels['16'] = 'Grasp and Lift'
mime_right_labels['17'] = 'Reach'
mime_right_labels['19'] = 'Reach'
mime_right_labels['22'] = 'Reach'
mime_right_labels['23'] = 'Grasp and Lift'
mime_right_labels['24'] = 'Reach'
mime_right_labels['26'] = 'Push'
mime_right_labels['27'] = 'Push'
mime_right_labels['32'] = 'Grasp and Lift'
mime_right_labels['33'] = 'Place'
mime_right_labels['35'] = 'Place'
mime_right_labels['38'] = 'Grasp and Lift'
mime_right_labels['39'] = 'Place'
mime_right_labels['40'] = 'Reach'
mime_right_labels['41'] = 'Reach'
mime_right_labels['42'] = 'Reach'
mime_right_labels['43'] = 'Reach'
mime_right_labels['44'] = 'Place'
mime_right_labels['46'] = 'Reach'
mime_right_labels['47'] = 'Reach'
mime_right_labels['49'] = 'Place'
mime_right_labels['50'] = 'Place'
mime_right_labels['51'] = 'Push'
mime_right_labels['52'] = 'Return'
mime_right_labels['55'] = 'Place'
mime_right_labels['56'] = 'Place'
mime_right_labels['57'] = 'Place'
mime_right_labels['58'] = 'Place'
mime_right_labels['61'] = 'Place'
mime_right_labels['66'] = 'Place'
mime_right_labels['70'] = 'Place and Return'
mime_right_labels['72'] = 'Reach'
mime_right_labels['82'] = 'Reach'
mime_right_labels['86'] = 'Return'
mime_right_labels['87'] = 'Place and Return'
mime_right_labels['88'] = 'Grasp and Lift'
mime_right_labels['94'] = 'Reach'
mime_right_labels['95'] = 'Reach'

mime_right_inverse_labels = {}
for k,v in mime_right_labels.items():
    print(k,v)
    if v not in mime_right_inverse_labels.keys():
        mime_right_inverse_labels[v] = [k]
    else:
        mime_right_inverse_labels[v].append(k)

