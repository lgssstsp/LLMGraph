
import os
import pandas as pd
import xml.etree.ElementTree as ET
import cairosvg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch, FancyArrowPatch
from matplotlib.path import Path


def node_level_draw_architecture(arch, save_path):
    print("========")
    print(arch.split("||"))
    components = arch.split("||")
    fig, ax = plt.subplots(figsize=(15, 9))
    nodes = [
        ('MLP', (0.5, 0.9), '#FFCE07'),
        (components[0], (0.15, 0.5), '#C5E0B4'),
        (components[1], (0.38, 0.5), '#C5E0B4'),
        (components[2], (0.61, 0.5), '#C5E0B4'), 
        (components[3], (0.84, 0.5), '#C5E0B4'),

        (components[-5], (0.15, 0.56), '#F4B183'),
        (components[-4], (0.38, 0.56), '#F4B183'),
        (components[-3], (0.61, 0.56), '#F4B183'),
        (components[-2], (0.84, 0.56), '#F4B183'),
        (components[-1], (0.5, 0.1), '#F4B183'),
        ('MLP', (0.5, 0.04), '#FFCE07'),

    ]

    width, height = 0.18, 0.06
    node_position = []
    node_position.append({"in": (nodes[0][1][0], nodes[0][1][1] + height / 2), "out": (nodes[0][1][0], nodes[0][1][1] - height / 2)})
    node_position.append({"in": (nodes[5][1][0], nodes[5][1][1] + height / 2), "out": (nodes[1][1][0], nodes[1][1][1] - height / 2)})
    node_position.append({"in": (nodes[6][1][0], nodes[6][1][1] + height / 2), "out": (nodes[2][1][0], nodes[2][1][1] - height / 2)})
    node_position.append({"in": (nodes[7][1][0], nodes[7][1][1] + height / 2), "out": (nodes[3][1][0], nodes[3][1][1] - height / 2)})
    node_position.append({"in": (nodes[8][1][0], nodes[8][1][1] + height / 2), "out": (nodes[4][1][0], nodes[4][1][1] - height / 2)})
    node_position.append({"in": (nodes[9][1][0], nodes[9][1][1] + height / 2), "out": (nodes[10][1][0], nodes[10][1][1] - height / 2)})

    for node, (x, y), color in nodes:
        rect_x, rect_y = x - width / 2, y - height / 2  
        rect = Rectangle((rect_x, rect_y), width, height, color=color, ec=None)
        ax.add_patch(rect)
        ax.text(x, y, node, color='black', fontsize=16, ha='center', va='center') 
    total_count = 0
    num_layer = 5
    for i in range(1,num_layer+1):
        for j in range(i):
            total_count += 1
            out_position =  node_position[j]['out']
            in_position =  node_position[i]['in']
            if components[total_count+num_layer-2] == 'identity':   
                flag = 1
                if flag == 1:
                    flag = 0
                    if out_position == node_position[0]['out'] or in_position == node_position[5]['in']:
                        conn = ConnectionPatch(xyA=out_position, xyB=in_position, color='black', coordsA='data', coordsB='data',arrowstyle='->', shrinkB=5)
                        ax.add_patch(conn)
                    else:
                        x1, y1 = out_position
                        x2, y2 = in_position
                        control_point1 = (x1 + 1/3 * abs(x2 - x1), y1 - abs(y2 - y1) * 1.3)  # 向上偏移
                        control_point2 = (x2 - 1/3 * abs(x2 - x1), y2 + abs(y2 - y1) * 1.3)  # 向下偏移
                        verts = [out_position, control_point1, control_point2, in_position]
                        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
                        path = Path(verts, codes)
                        patch = FancyArrowPatch(path=path, arrowstyle='->', color='black', linewidth=1, mutation_scale=10)
                        ax.add_patch(patch)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.show()
    plt.savefig(save_path)
    plt.close(fig)



def graph_level_draw_architecture(arch, save_path, num_layer):
    components = arch.split("||")

    fig, ax = plt.subplots(figsize=(15, 30))
    width, height = 0.14, 0.02
    nodes = []
    nodes.append(('MLP', (0.5, 0.98), '#FFCE07'))
    gap = 0.96/(num_layer+3)
    for i in range(num_layer):
        nodes.append((components[i], (0.5, 0.98 - (i+1) * gap), '#C5E0B4'))
    for i in range(num_layer):
        nodes.append((components[-(2+num_layer-i)], (0.5, 0.98 - (i+1) * gap + height), '#F4B183'))
    nodes.append((components[-2], (0.5, 0.98 - (num_layer+2) * gap + height), '#F4B183'))
    nodes.append((components[-1], (0.5, 0.98 - (num_layer+2) * gap), '#DDE3F2'))
    nodes.append(('MLP', (0.5, 0.98 - (num_layer+3) * gap), '#FFCE07'))

    node_position = []
    node_position.append({"in": (nodes[0][1][0], nodes[0][1][1] + height / 2), "out": (nodes[0][1][0], nodes[0][1][1] - height / 2)})

    for i in range(num_layer):
        node_position.append({"in": (nodes[i+1+num_layer][1][0], nodes[i+1+num_layer][1][1] + height / 2), "out": (nodes[i+1][1][0], nodes[i+1][1][1] - height / 2)})

    node_position.append({"in": (nodes[-3][1][0], nodes[-3][1][1] + height / 2), "out": (nodes[-2][1][0], nodes[-2][1][1] - height / 2)})

    
    node_position.append({"in": (nodes[-1][1][0], nodes[-1][1][1] + height / 2), "out": (nodes[-1][1][0], nodes[-1][1][1] - height / 2)})

    for node, (x, y), color in nodes:
        rect_x, rect_y = x - width / 2, y - height / 2 
        rect = Rectangle((rect_x, rect_y), width, height, color=color, ec=None)
        ax.add_patch(rect)
        ax.text(x, y, node, color='black', fontsize=16, ha='center', va='center')  


    total_count = 0
    num_layer = num_layer + 1
    for i in range(1,num_layer+1):
        for j in range(i):
            total_count += 1
            out_position =  node_position[j]['out']
            in_position =  node_position[i]['in']
            if components[total_count+num_layer-2] == 'identity':             
                
                flag = 1
                if flag == 1:
                    flag = 0

                    if i == j+1:
                        conn = ConnectionPatch(xyA=out_position, xyB=in_position, color='black', coordsA='data', coordsB='data',arrowstyle='->', shrinkB=5)
                        ax.add_patch(conn)
                    else :
                        x1, y1 = out_position
                        x2, y2 = in_position
                        offset = (i-j)*(gap)
                        if i % 2 == 0:
                            control_point = (x1 + offset, (y1 + y2)/2) 
                        else:
                            control_point = (x1 - offset, (y1+y2)/2) 


                        verts = [out_position, control_point, in_position]
                        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                        path = Path(verts, codes)

                        patch = FancyArrowPatch(path=path, arrowstyle='->', color='black', linewidth=1, mutation_scale=10)
                        ax.add_patch(patch)
    
    out_position =  node_position[-2]['out']
    in_position =  node_position[-1]['in']
    conn = ConnectionPatch(xyA=out_position, xyB=in_position, color='black', coordsA='data', coordsB='data',arrowstyle='->', shrinkB=5)
    ax.add_patch(conn)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.show()
    plt.savefig(save_path)
    plt.close(fig)



def link_level_draw_architecture(arch, save_path):

    tree = ET.parse('../..//link.svg')  
    root = tree.getroot()
    namespaces = {'svg': 'http://www.w3.org/2000/svg'}
    

    for key, value in arch.items():
        for text in root.findall('.//svg:text', namespaces):
            if text.text:
                modified_text = text.text.replace(key, key + ":" + str(value))
                text.text = modified_text

    modified_svg_path = 'modified_link_svg.svg'
    tree.write(modified_svg_path)

    with open(modified_svg_path, 'r') as svg_file:
        svg_code = svg_file.read()
        cairosvg.svg2png(bytestring=svg_code, write_to=save_path, dpi=1000)

