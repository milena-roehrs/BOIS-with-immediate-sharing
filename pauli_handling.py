import copy

def twopaulis(list1, list2):
    """
    Compares two lists and creates a new list that combines the unique elements from both lists.
    Additionally, it returns a mapping list that indicates the index of each element in the new list.

        Parameters:
            list1 (list): The first list to be compared.
            list2 (list): The second list to be compared.

        Returns:
            tuple: A tuple containing two elements:
                - list3 (list): The new list that combines the unique elements from both input lists.
                - mapping (list): A list of indices that maps the elements in list2 to their corresponding
                positions in list3.
    """

    mapping = []
    list3 = copy.deepcopy(list1)
    for wl in list2:
        found = False
        indx = 0
        laenge = len(list3)
        while (not found) and (indx < laenge):
            if wl == list3[indx]:
                found = True
                mapping.append(indx)
            else:
                indx += 1

        if (not found):
            list3.append(wl)
            mapping.append(laenge)

    return list3, mapping