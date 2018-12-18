import csv

width = 800
height = 1200
ofs = {'TOP': 0, 'BOTTOM': 600}

per_frame = False

with open('..\default_node_pos.csv') as infile, open('..\overlay_vstack_rel.csv', 'w', newline='') as outfile:
    reader = csv.reader(infile)
    header = next(reader)

    writer = csv.writer(outfile)
    writer.writerow(header)

    for row in reader:
        row[1] = row[1].strip()

        # Exact node position
        row[2] = '{:.3f}'.format(int(row[2]) / width)
        if per_frame:
            row[3] = '{:.3f}'.format((int(row[3]) - ofs[row[0].strip()]) / (height - ofs[row[0].strip()]))
        else:
            row[3]  = '{:.3f}'.format(int(row[3]) / height)

        # Label postition
        row[4] = '{:.3f}'.format(int(row[4]) / width)
        if per_frame:
            row[5] = '{:.3f}'.format((int(row[5]) - ofs[row[0].strip()]) / (height - ofs[row[0].strip()]))
        else:
            row[5]  = '{:.3f}'.format(int(row[5]) / height)

        writer.writerow(row)
