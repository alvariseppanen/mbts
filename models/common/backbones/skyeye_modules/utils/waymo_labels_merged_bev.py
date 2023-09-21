#!/usr/bin/python
#
# KITTI-360 labels
#

from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'waymoId'     , # An integer ID that is associated with this label for KITTI-360
                    # NOT FOR RELEASING

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                                     id    waymoId     trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'                           ,  0 ,       0 ,       255 ,     'void'            , 0       , False          , True      , (  0,  0,  0) ),
    Label(  'road'                                ,  1 ,      18 ,         0 ,     'flat'            , 6       , False          , False      , (128,  64, 128) ),
    Label(  'lane_marker'                         ,  2 ,      19 ,         0 ,     'flat'            , 7       , False          , False     , (128,  64, 128) ),
    Label(  'other_ground'                        ,  3 ,      20 ,         0 ,     'flat'            , 7       , False          , False     , (128,  64, 128) ),
    Label(  'sidewalk'                            ,  4 ,      22 ,         1 ,     'flat'            , 7       , False          , False     , (244,  35, 232) ),
    Label(  'curb'                                ,  5 ,      17 ,         1 ,     'flat'            , 6       , False          , True      , (244,  35, 232) ),
    Label(  'walkable'                            ,  6 ,      21 ,       255 ,     'flat'            , 7       , False          , False     , (  99, 146, 168) ),
    Label(  'building'                            ,  7 ,      14 ,         2 ,     'construction'    , 6       , False          , False     , ( 70,  70,  70) ),
    Label(  'traffic_sign'                        ,  8 ,       8 ,       255 ,     'object'          , 4       , False          , False     , (152, 251, 152) ),
    Label(  'traffic_light'                       ,  9 ,       9 ,       255 ,     'object'          , 5       , False          , True      , ( 70, 130, 180) ),
    Label(  'pole'                                , 10 ,      10 ,       255 ,     'object'          , 10      , False          , False     , (140, 140, 140) ),
    Label(  'construction_cone'                   , 11 ,      11 ,       255 ,     'object'          , 6       , False          , True     , (220,  20,  60) ),
    Label(  'vegetation'                          , 12 ,      15 ,       255 ,     'nature'          , 6       , False          , True     , (107, 142,  35) ),
    Label(  'tree_trunk'                          , 13 ,      16 ,       255 ,     'nature'          , 6       , False          , True      , (107, 142,  35) ),
    Label(  'pedestrian'                          , 14 ,       7 ,         3 ,     'human'           , 4       , True           , False     , (220, 20,  60) ),
    Label(  'car'                                 , 15 ,       1 ,         4 ,     'vehicle'         , 0       , True           , False      , (  0,  0, 142) ),
    Label(  'truck'                               , 16 ,       2 ,         4 ,     'large_vehicle'   , 0       , True           , False      , (  0, 60, 100) ),
    Label(  'bus'                                 , 17 ,       3 ,         4 ,     'large_vehicle'   , 0       , True           , False      , (  0, 60, 100) ),
    Label(  'motorcyclist'                        , 18 ,       5 ,         5 ,     'human'           , 1       , True           , False     , (119, 11,  32) ),
    Label(  'bicyclist'                           , 19 ,       6 ,         5 ,     'human'           , 2       , True           , False     , (119, 11,  32) ),
    Label(  'bicycle'                             , 20 ,      12 ,         5 ,     'vehicle'         , 6       , True           , False     , (119,  11,  32) ),
    Label(  'motorcycle'                          , 21 ,      13 ,         5 ,     'vehicle'         , 6       , True           , False     , (119,  11,  32) ),
    Label(  'other_vehicle'                       , 22 ,       4 ,       255 ,     'vehicle'         , 1       , True           , False     , (  0,  0,   0) ),
]

#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# NuScenes ID to cityscapes ID
waymoId2label   = { label.waymoId : label for label in labels           }
# Stuff and things
stuff_classes   = [label.id for label in labels if not label.hasInstances and not label.ignoreInEval]
thing_classes   = [label.id for label in labels if label.hasInstances and not label.ignoreInEval]
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]

#--------------------------------------------------------------------------------
# Assure single instance name
#--------------------------------------------------------------------------------

# returns the label name that describes a single instance (if possible)
# e.g.     input     |   output
#        ----------------------
#          car       |   car
#          cargroup  |   car
#          foo       |   None
#          foogroup  |   None
#          skygroup  |   None
def assureSingleInstanceName( name ):
    # if the name is known, it is not a group
    if name in name2label:
        return name
    # test if the name actually denotes a group
    if not name.endswith("group"):
        return None
    # remove group
    name = name[:-len("group")]
    # test if the new name exists
    if not name in name2label:
        return None
    # test if the new name denotes a label that actually has instances
    if not name2label[name].hasInstances:
        return None
    # all good then
    return name

#--------------------------------------------------------------------------------
# Main for testing
#--------------------------------------------------------------------------------

# just a dummy main
if __name__ == "__main__":
    # Print all the labels
    print("List of Waymo Open Dataset labels:")
    print("")
    print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( 'name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval' ))
    print("    " + ('-' * 98))
    for label in labels:
        # print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( label.name, label.id, label.trainId, label.category, label.categoryId, label.hasInstances, label.ignoreInEval ))
        print(" \"{:}\"".format(label.name))
    print("")

    print("Example usages:")

    # Map from name to label
    name = 'car'
    id   = name2label[name].id
    print("ID of label '{name}': {id}".format( name=name, id=id ))

    # Map from ID to label
    category = id2label[id].category
    print("Category of label with ID '{id}': {category}".format( id=id, category=category ))

    # Map from trainID to label
    trainId = 0
    name = trainId2label[trainId].name
    print("Name of label with trainID '{id}': {name}".format( id=trainId, name=name ))