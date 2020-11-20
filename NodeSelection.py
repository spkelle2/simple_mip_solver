class NodeSelection_BestFirst():
    def __init__(self):
        pass

    def choose(self, nodes_list):
        return nodes_list[0]


NodeSelections = {'BestFirst': NodeSelection_BestFirst}
