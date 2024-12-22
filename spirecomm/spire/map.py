class Node:

    def __init__(self, x, y, symbol):
        self.x = x
        self.y = y
        self.symbol = symbol
        self.children = []
        name = ''
        if self.symbol == "M":
            name = "Enemy"
        elif self.symbol == "?":
            name = "Unknown"
        elif self.symbol == "$":
            name = "Merchant"
        elif self.symbol == "E":
            name = "Elite"
        elif self.symbol == "R":
            name = "Rest"
        elif self.symbol == "T":
            name = "Treasure"
        else:
            name = ""
        self.name = name

    @classmethod
    def from_json(cls, json_object):
        return cls(json_object.get("x"), json_object.get("y"), json_object.get("symbol"))

    def __repr__(self):

        return "{}({},{})".format(self.name,self.x, self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def to_dict(self, max_depth=None, current_depth=0):
        # 如果当前深度超过最大深度，返回一个包含当前节点信息的字典
        if max_depth is not None and current_depth >= max_depth:
            return {
                "x": self.x,
                "y": self.y,
                "name": self.name,
                "children": []
            }

        # 创建一个字典表示当前节点
        node_dict = {
            "x": self.x,
            "y": self.y,
            "name": self.name,
            "children": [child.to_dict(max_depth, current_depth + 1) for child in self.children]
        }
        return node_dict

class Map:

    def __init__(self):
        self.nodes = {}

    def add_node(self, node):
        if node.y in self.nodes:
            self.nodes[node.y][node.x] = node
        else:
            self.nodes[node.y] = {node.x: node}

    def get_node(self, x, y):
        if y in self.nodes and x in self.nodes[y]:
            return self.nodes[y][x]
        else:
            return None

    @classmethod
    def from_json(cls, node_list):
        dungeon_map = Map()
        for json_node in node_list:
            node = Node.from_json(json_node)
            dungeon_map.add_node(node)

        for json_node in node_list:
            children = json_node.get("children")
            parent_node = dungeon_map.get_node(json_node.get("x"), json_node.get("y"))
            for json_child in children:
                child_node = dungeon_map.get_node(json_child.get("x"), json_child.get("y"))
                if child_node is not None:
                    parent_node.children.append(child_node)

        return dungeon_map
