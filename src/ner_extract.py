


def extract_ner_conll(path, sep="\t", mapper = {}):
    #Build sentences with tag sequence
    # from: 3	Guatemala	Guatemala	subst	subst	prop	2	PUTFYLL	_	name=B-GPE_LOC
    doc = [] # List of dicts, idx, tokens, tags
    #Per sentence:
    tokens = []
    tags = []
    idx = ""
    # returns list of tuples with two lists
    with open (path, 'r') as file:
      data = file.readlines()
    for line in data:
      line = line.strip()
      if len(line) == 0: # End of sentence
        if len(tokens) > 0:
          doc.append({"idx": idx, "tokens":tokens, "ner_tags":tags})
        tokens = []
        tags = []
      else: # Line not empty
        if line[0] == '#':
          if "sent_id" in line: #Get new sent_id
            idx = line.split("=")[-1].strip()
        else:
            line = line.split(sep)
            tokens.append(line[1])
            ner_field = line[-1] # name=B-ORG
            assert "name=" in ner_field
            ner_tag = ner_field.split("=")[-1].strip()
            ner_tag = mapper.get(ner_tag, ner_tag)
            tags.append(ner_tag)
          
    return doc