import graphviz 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import _pickle as pickle
import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'

# Draw neural network with arbitrary topology.
def draw_net(net, filename=None, is_3d = False):
    node_names = {}
    node_colors = {}

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph('svg', node_attr=node_attrs)

    inputs = set()
    for k in net.input_nodes:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 
                       'shape': 'box', 
                       'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in net.output_nodes:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 
                      'fillcolor': node_colors.get(k, 'lightblue')}
        dot.node(name, _attributes=node_attrs)

    for node, act_func, agg_func, bias, response, links in net.node_evals:
        for i, w in links:
            input, output = node, i
            a = node_names.get(output, str(output))
            b = node_names.get(input, str(input))
            style = 'solid'
            color = 'green' if w > 0.0 else 'red'
            width = str(0.1 + abs(w / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename)

    return dot


# Click handler for weight gradient created by a CPPN. Will re-query with the clicked coordinate.
def onclick(event):
    plt.close()
    x = event.xdata
    y = event.ydata
    
    path_to_cppn = "es_hyperneat_xor_small_cppn.pkl"
    with open(path_to_cppn, 'rb') as input:  # For now, path_to_cppn should match path in test_cppn.py, sorry.
        cppn = pickle.load(input)
        from pureples.es_hyperneat.es_hyperneat import find_pattern
        pattern = find_pattern(cppn, (x, y))
        draw_pattern(pattern)


# Draws the pattern/weight gradient queried by a CPPN. 
def draw_pattern(im, res=60):
    fig = plt.figure()
    plt.axis([-1, 1, -1, 1])
    fig.add_subplot(111)

    a = range(res)
    b = range(res)

    for x in a:
        for y in b:
            px = -1.0 + (x/float(res))*2.0+1.0/float(res)
            py = -1.0 + (y/float(res))*2.0+1.0/float(res)
            c = str(0.5-im[x][y]/float(res))
            plt.plot(px, py, marker='s', color=c)

    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.grid()
    plt.show()


# Draw the net created by ES-HyperNEAT
def draw_es(id_to_coords, connections, filename):
    fig = plt.figure()
    end_x = 0
    end_y = 0
    end_z = 0
    for (coord, idx) in id_to_coords.items():
        if(coord[0] > end_x):
            end_x = coord[0]
        if(coord[1] > end_y):
            end_y = coord[1]
    plt.axis([-end_x, end_x, -end_y, end_y])
    fig.add_subplot(111)

    for c in connections:
        color = 'red'
        if c.weight > 0.0:
            color = 'black'
        plt.arrow(c.x1, c.y1, c.x2-c.x1, c.y2-c.y1, head_width=0.00, head_length=0.0, 
                  fc=color, ec=color, length_includes_head = True)

    for (coord, idx) in id_to_coords.items():
        plt.plot(coord[0], coord[1], marker='o', markersize=8.0, color='grey')

    plt.grid()
    fig.savefig(filename)
    
def draw_es_nd(id_to_coords, connections, filename):
    fig = plt.figure()
    end_x = 0
    end_y = 0
    end_z = 0
    xs = []
    ys = []
    zs = []
    for (coord, idx) in id_to_coords.items():
        xs.append(coord[0])
        ys.append(coord[1])
        zs.append(coord[2])

    #plt.axis([-end_x, end_x, -end_y, end_y, -end_z, end_z])
    ax = fig.gca(projection='3d')
    
    for c in connections:
        xc = (c.coord1[0], c.coord2[0])
        yc = (c.coord1[1], c.coord2[1])
        zc = (c.coord1[2], c.coord2[2])
        color = 'r'
        if c.weight > 0.0:
            color = 'g'
        ax.plot3D(xc, yc, zc, color='g')
    
    ax.scatter3D(xs, ys, zs)
    #ax.plot((xs[0],ys[0],zs[0]), (xs[-1],ys[-1],zs[-1]), color='g')
    '''
    for (coord, idx) in id_to_coords.items():
        plt.plot(coord[0], coord[1], marker='o', markersize=8.0, color='grey')
    '''
    fig.savefig(filename)
    plt.show()
    fig.savefig(filename)
    plt.close()
    

