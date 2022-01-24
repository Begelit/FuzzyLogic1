import csv
import os
import os.path
from scipy.integrate import quad
from matplotlib import pyplot as plt
import numpy as np

def searchMinMax(header,filename):
	with open(filename, encoding='utf-8-sig') as heart_f:
		file_reader = csv.reader(heart_f, delimiter = ",")
		minVal = 1000
		maxVal = 0
		headers = next(file_reader)
		for row in file_reader:
			if row[headers.index('sex')] == '1': #and row[headers.index('oldpeak')]:
				if float(row[headers.index(header)]) <= minVal:
					minVal = float(row[headers.index(header)])
				elif float(row[headers.index(header)]) >= maxVal:
					maxVal = float(row[headers.index(header)])
		return {'header':header,'min':minVal,'max':maxVal,
				'M2': minVal,'D2':maxVal,
				'S': minVal + (maxVal - minVal)/2,
				'M1': minVal + (minVal + (maxVal - minVal)/2 - minVal)/2,
				'D1': minVal + (maxVal - minVal)/2 + (maxVal - (minVal + (maxVal - minVal)/2))/2}
	
		
def membershipFunc(value,minmaxDataDict):
	minVal = minmaxDataDict['min']
	maxVal = minmaxDataDict['max']
	m2 = minmaxDataDict['M2']
	m1 = minmaxDataDict['M1']
	s = minmaxDataDict['S']
	d1 = minmaxDataDict['D1']
	d2 = minmaxDataDict['D2']
	if value >= m2 and value <= m1:
		return [str(round(( m1 - value )/( m1 - m2 ),2)),
				 str(round(( value - m2 )/( m1 - m2 ),2)),
				 '0','0','0']
	elif value >= m1 and value <= s:
		return ['0',
				 str(round(( s - value )/( s - m1 ),2)),
				 str(round(( value - m1 )/( s - m1 ),2)),
				 '0','0']
	elif value >= s and value <= d1:
		return ['0','0',
				 str(round(( d1 - value )/( d1 - s ),2)),
				 str(round((value-s)/(d1-s),2)),
				 '0']
	elif value >= d1 and value <= d2:
		return ['0','0','0',
				 str(round(( d2 - value )/( d2 - d1 ),2)),
				 str(round(( value - d1 )/( d2 - d1 ),2))]
	else: return ['0','0','0','0','0']

def writeMembershipResults(inputValheader,readingFileName):
	with open(readingFileName, encoding='utf-8-sig') as r_f:
		file_reader = csv.reader(r_f,delimiter = ",")
		headers = next(file_reader)
		data = [[inputValheader,'M2','M1','S','D1','D2']]
		for row in file_reader:
			if row[headers.index('sex')] == '1': #and row[headers.index('oldpeak')] != '5.6' :
				data = data+[[row[headers.index(inputValheader)]]+membershipFunc(float(row[headers.index(inputValheader)]),
																				 searchMinMax(inputValheader,readingFileName))]
		return data

def searchMinMaxOut(header, filename):
	with open(filename, encoding='utf-8-sig') as heart_f:
		file_reader = csv.reader(heart_f, delimiter = ",")
		minVal = 10
		maxVal = 0
		headers = next(file_reader)
		for row in file_reader:
			if row[headers.index('sex')] == '1': #and row[headers.index('oldpeak')] != '5.6':
				if float(row[headers.index(header)]) <= minVal:
					minVal = float(row[headers.index(header)])
				elif float(row[headers.index(header)]) >= maxVal:
					maxVal = float(row[headers.index(header)])
		return {'header':header,'min':minVal,'max':maxVal,'M':minVal,'S':minVal + (maxVal - minVal)/2,'D':maxVal}

def membershipFuncOut(value,minmaxDataDict):
	minVal = minmaxDataDict['min']
	maxVal = minmaxDataDict['max']
	m = minmaxDataDict['M']
	s = minmaxDataDict['S']
	d = minmaxDataDict['D']	
	if value >= m and value <= s:
		return [str(round((s-value)/(s-m),2)), str(round((value-m)/(s-m),2)),'0']
	elif value >= s and value <= d:
		return ['0',str(round((d-value)/(d-s),2)),str(round((value-s)/(d-s),2))]
				
def writeMembershipResultsOut(outputValheader,readingFileName):
	with open(readingFileName, encoding='utf-8-sig') as r_f:	
		file_reader = csv.reader(r_f,delimiter = ",")
		headers = next(file_reader)
		dataOut=[[outputValheader, 'M', 'S','D']]		
		for row in file_reader:
			if row[headers.index('sex')] == '1':# and row[headers.index('oldpeak')] != '5.6' :
				dataOut = dataOut+[[row[headers.index(outputValheader)]]+membershipFuncOut(float(row[headers.index(outputValheader)]),
																						   searchMinMaxOut(outputValheader,readingFileName))]
		return dataOut

def promDatafunc(promData,row):
	n = 0
	for rowN in promData:
		if row[0:5] == rowN[0:5]:
			n+=1
	if n == 0:
		return True
	
def deleteRegDuplicate(data):
	promData1 = []
	promData2 = []
	n = 0;
	for row1 in data:
		if promDatafunc(promData1,row1) == True:
			promData1+=[row1]
			maxSP = row1
			for row2 in data:
				if maxSP[0:5] == row2[0:5]:
					promData1+=[row2]
					if maxSP[5] < row2[5]:
						maxSP = row2
			promData2+=[maxSP]
			n+=1
	return promData2

def writingResultsInCSV(data, fileName):					
	if os.path.exists('/home/koza/NL/1/asd/male/Results/'+fileName + '.csv') == True:
		os.remove('/home/koza/NL/1/asd/male/Results/'+fileName + '.csv')	
	with open('/home/koza/NL/1/asd/male/Results/'+fileName+'.csv', 'w', encoding = 'utf-8') as w_f:
		writer = csv.writer(w_f)
		for row in data:
			writer.writerow(row)
			
	
def createRegulationBase(headers,outputHeaders,readingFile):
	dictMembership = dict()
	for row in headers:
		dictMembership[row] = writeMembershipResults(row,readingFile)
		writingResultsInCSV(dictMembership[row],row + 'MembershipResults')
	dictMembership[outputHeaders] = writeMembershipResultsOut(outputHeaders,readingFile)
	writingResultsInCSV(dictMembership[outputHeaders],outputHeaders + 'MembershipResultsOutp')
	allData = []
	for i in range(1,len(dictMembership[headers[0]])):
		promData = []
		SP = 1
		r=1
		for row in headers:
			headerdata = dictMembership[row][0]
			SP = SP * float(max(dictMembership[row][i][1:]))
			r = r * float(max(dictMembership[row][i][1:]))
			promData = promData + [headerdata[dictMembership[row][i].index( max(dictMembership[row][i][1:]) )]]
		SP = SP * float(max(dictMembership[outputHeaders][i][1:]))
		headerdataout = dictMembership[outputHeaders][0]
		promData = promData + [headerdataout[dictMembership[outputHeaders][i].index( max(dictMembership[outputHeaders][i][1:]) )]]+[str(round(SP,3))]+[str(round(r,3))]
		allData = allData + [promData]
	writingResultsInCSV(allData, 'FullRegulationBase')
	writingResultsInCSV(deleteRegDuplicate(allData), 'RegulationBaseWithoutDuplicate')
	dictAllData = dict()
	dictAllData['data'] = deleteRegDuplicate(allData)
	dictAllData['headers'] = headers + [outputHeaders] + ['SP']+['SK']
	return dictAllData

#print(createRegulationBase(['age','trestbps','chol','thalach'],'oldpeak','heart.csv'))

def selectFuncMem(group, header, value):
	if group == 'M2':
		return membershipFunc(value,searchMinMax(header,'heart.csv'))[0]
	elif group == 'M1':
		return membershipFunc(value,searchMinMax(header,'heart.csv'))[1]
	elif group == 'S':
		return membershipFunc(value,searchMinMax(header,'heart.csv'))[2]
	elif group == 'D1':
		return membershipFunc(value,searchMinMax(header,'heart.csv'))[3]
	elif group == 'D2':
		return membershipFunc(value,searchMinMax(header,'heart.csv'))[4]
	
def calculateY(variableStr,outpGroupStr, minMemValue,count):
	searchMMData = searchMinMaxOut(variableStr,'heart.csv')
	M = searchMMData['M']
	S = searchMMData['S']
	D = searchMMData['D']
	MS = minMemValue*S-minMemValue*M-M 
	DS = D-minMemValue*(S-M)#D-minMemValue*M+minMemValue*S 

	#<<<<plot
	fig,ax = plt.subplots()
	plt.axhline(y=minMemValue, color = 'red', linewidth = 4)
	
	ax.minorticks_on()

	ax.grid(which='major',color = 'k',linewidth = 2)
	ax.grid(which='minor', color = 'k', linestyle = ':')
	ax.set_xlim([M, D])
	ax.set_ylim([0, 1])
	ax.scatter(M,minMemValue,linewidth = 4)
	plt.annotate('minFP = '+str(minMemValue), xy=(M,minMemValue),xycoords='data',
				 xytext = (M,minMemValue+minMemValue/40),textcoords='data',fontweight='bold',fontsize=12,color = 'red' )
	ax.set_title(str(count)+')',loc='left',fontsize=25,fontweight='bold',color='b')
	
	#plot>>>>
	
	def integFuncMS(x):
		return x*((x-M)/(S-M))
	def integFuncS(x):
		return x*minMemValue
	def integFuncDS(x):
		return x*((D-x)/(D-S))
	
	if outpGroupStr == 'M':
		
		#<<<<plot
		x2 = np.linspace(M,S)
		y2 = (S-x2)/(S-M)
		ax.plot(x2,y2,'b',linewidth = 3)
		ax.scatter(S - minMemValue*(S-M),minMemValue,linewidth = 4)
		plt.xlabel('M', fontsize=20, fontweight='bold')
		plt.axvline(x=S - minMemValue*(S-M), color = 'm', linewidth = 4,linestyle = '--', ymax=minMemValue)
		plt.annotate('Y = '+str(round(S - minMemValue*(S-M),3)), xy=(S - minMemValue*(S-M),minMemValue),xycoords='data',
					 xytext = (S - minMemValue*(S-M),minMemValue+minMemValue/40),textcoords='data',fontweight='bold',fontsize=14,color = 'm' )
		if os.path.exists('/home/koza/NL/1/asd/male/Plots/Deffuz/' +str(count)+ '.png') == True:
			os.remove('/home/koza/NL/1/asd/male/Plots/Deffuz/' +str(count)+ '.png')
		plt.savefig('/home/koza/NL/1/asd/male/Plots/Deffuz/' +str(count)+ '.png')
		#plt.show()
		#plot>>>>
		
		return S - minMemValue*(S-M)
	
	elif outpGroupStr == 'D':
		
		#<<<<plot
		x2 = np.linspace(S,D)
		y2 = (x2-S)/(D-S)
		ax.plot(x2,y2,'y',linewidth = 3)
		ax.scatter(minMemValue*(D-S)+S,minMemValue,linewidth = 4)
		plt.xlabel('D', fontsize=20, fontweight='bold')
		plt.axvline(x=minMemValue*(D-S)+S, color = 'm', linewidth = 4,linestyle = '--', ymax=minMemValue)
		plt.annotate('Y = '+str(round(minMemValue*(D-S)+S,3)), xy=(minMemValue*(D-S)+S,minMemValue),xycoords='data',
					 xytext = (minMemValue*(D-S)+S,minMemValue+minMemValue/40),textcoords='data',fontweight='bold',fontsize=14,color = 'm' )
		if os.path.exists('/home/koza/NL/1/asd/male/Plots/Deffuz/' +str(count)+ '.png') == True:
			os.remove('/home/koza/NL/1/asd/male/Plots/Deffuz/' +str(count)+ '.png')
		plt.savefig('/home/koza/NL/1/asd/male/Plots/Deffuz/' +str(count)+ '.png')
#		plt.show()
		#plot>>>>

		return minMemValue*(D-S)+S
	
	else:		
		squareTrapez = 0.5*(DS-MS+searchMMData['max']-searchMMData['min'])*minMemValue
		integral = (quad(integFuncMS,M,MS)[0]+quad(integFuncS,MS,DS)[0]+quad(integFuncDS,DS,D)[0])
		
		#<<<<plot
		x3 = np.linspace(M,S)
		y3 = (x3-M)/(S-M)
		x4 = np.linspace(S,D)
		y4 = (D-x4)/(D-S)
		ax.plot(x3,y3,'g',linewidth = 3)
		ax.plot(x4,y4,'g',linewidth = 3)
		xAll = np.linspace(M,D)
		plt.fill_between(x3, y3,where=y3<=minMemValue,color='m')
		plt.fill_between(x4, y4,where=y4<=minMemValue,color='m')
		plt.xlabel('S', fontsize=20, fontweight='bold')

		ax.axvspan(MS,DS,0,minMemValue,color='m')

		plt.annotate('square = '+str(round(squareTrapez,3)), xy=(0.02,0.95),xycoords='data',
					 xytext = (0.02,0.95),textcoords='data',fontweight='bold',fontsize=13,color = 'm' )
		plt.annotate('integral(x*FP(x)*dx) = '+str(round(integral,3)), xy=(0.02,0.9),xycoords='data',
					 xytext = (0.02,0.9),textcoords='data',fontweight='bold',fontsize=13,color = 'm' )
		plt.annotate('Y = integral(x*FP(x)*dx)/square =  '+str(round(integral/squareTrapez,3)), xy=(0.02,0.85),xycoords='data',
					 xytext = (0.02,0.85),textcoords='data',fontweight='bold',fontsize=13,color = 'm' )
		if os.path.exists('/home/koza/NL/1/asd/male/Plots/Deffuz/' +str(count)+ '.png') == True:
			os.remove('/home/koza/NL/1/asd/male/Plots/Deffuz/' +str(count)+ '.png')
		plt.savefig('/home/koza/NL/1/asd/male/Plots/Deffuz/' +str(count)+ '.png')
#		plt.show()
		#plot>>>>
		
		return integral/squareTrapez
	
def defuzzification(data,value):
	regulBase = []
	headers = data['headers'][0:4]
	znam = 0
	chisl = 0
	count = 0
	promRegulBase = []
	for row1 in data['data']:
		promRegulBase1 = [str(count)]
		n=0
		minFMem = 1
		for row2 in row1[0:4]:
			valueRow = selectFuncMem(row2, headers[n], value[headers[n]])
			if float(valueRow) <= minFMem :
				minFMem = float(valueRow)		
#			promRegulBase = promRegulBase + [{ headers[n]:selectFuncMem(row2, headers[n], value[headers[n]]), 'group': row2 }]
			promRegulBase1 += [headers[n], row2 ,selectFuncMem(row2, headers[n], value[headers[n]])]
			n+=1
		#print(data['headers'][4], row1[4],minFMem)
		if minFMem > 0:
#			promRegulBase += [{data['headers'][4]:minFMem,'group':row1[4],'Y':calculateY(data['headers'][4],row1[4], minFMem),'SP':row1[6]}]
			calcY = round(calculateY(data['headers'][4],row1[4], minFMem,count),3)
			promRegulBase1 += [data['headers'][4],row1[4],'minFP',minFMem,'Y',str(calcY),'SP',row1[6]]
			promRegulBase += [promRegulBase1]

			chisl+= calcY*float(row1[6])
			znam+=float(row1[6])
			print(promRegulBase1)
			count+=1

	writingResultsInCSV(promRegulBase, 'defuz')
	print(round(chisl/znam,3))

def createPlotsMembersFunc(minmaxDataDict):
	m2 = minmaxDataDict['M2']
	m1 = minmaxDataDict['M1']
	s = minmaxDataDict['S']
	d1 = minmaxDataDict['D1']
	d2 = minmaxDataDict['D2']
	fig,ax = plt.subplots()

	x1 = np.linspace(m2,m1)
	yM2 = ( m1 - x1 )/( m1 - m2 )
	yM1 = ( x1 - m2 )/( m1 - m2 )
	x2 = np.linspace(m1,s)
	yM12 = ( s - x2 )/( s - m1 )
	yS1 = ( x2 - m1 )/( s - m1 )
	x3 = np.linspace(s,d1)
	yS2 = ( d1 - x3 )/( d1 - s )	
	yD1 = ( x3 - s )/( d1 - s )
	x4 = np.linspace(d1,d2)
	yD12 = ( d2 - x4 )/( d2 - d1 )
	yD2 = ( x4 - d1 )/( d2 - d1 )

	ax.plot(x1, yM2, 'r')
	ax.plot(x1, yM1, 'y')
	ax.plot(x2, yM12, 'y')
	ax.plot(x2, yS1, 'b')
	ax.plot(x3, yS2, 'b')
	ax.plot(x3, yD1, 'm')
	ax.plot(x4, yD12, 'm')
	ax.plot(x4, yD2, 'g')

	ax.minorticks_on()

	ax.grid(which='major',color = 'k',linewidth = 2)
	ax.grid(which='minor', color = 'k', linestyle = ':')
    
	plt.xlabel(minmaxDataDict['header'], fontsize=14, fontweight='bold')

	plt.annotate('M2 = '+str(minmaxDataDict['M2']), xy=(minmaxDataDict['M2'],1),xycoords='data', xytext = (minmaxDataDict['M2'],1.03),textcoords='data',fontweight='bold',fontsize=9 )
	plt.annotate('M1 = '+str(minmaxDataDict['M1']), xy=(minmaxDataDict['M1'],1),xycoords='data', xytext = (minmaxDataDict['M1'],1.03),textcoords='data',fontweight='bold',fontsize=9 )
	plt.annotate('S = '+str(minmaxDataDict['S']), xy=(minmaxDataDict['S'],1),xycoords='data', xytext = (minmaxDataDict['S'],1.03),textcoords='data',fontweight='bold',fontsize=9 )
	plt.annotate('D1 = '+str(minmaxDataDict['D1']), xy=(minmaxDataDict['D1'],1),xycoords='data', xytext = (minmaxDataDict['D1'],1.03),textcoords='data',fontweight='bold',fontsize=9 )
	plt.annotate('D2 = '+str(minmaxDataDict['D2']), xy=(minmaxDataDict['D2'],1),xycoords='data', xytext = (minmaxDataDict['D2'],1.03),textcoords='data',fontweight='bold',fontsize=9 )
	if os.path.exists('/home/koza/NL/1/asd/male/Plots/MembFunc/'+minmaxDataDict['header']+'.png') == True:
		os.remove('/home/koza/NL/1/asd/male/Plots/MembFunc/'+minmaxDataDict['header']+'.png')
	plt.savefig('/home/koza/NL/1/asd/male/Plots/MembFunc/'+minmaxDataDict['header']+'.png')
#	plt.show()

def createPlotsMembersFuncOut(minmaxDataDictOut):
	m = minmaxDataDictOut['M']
	s = minmaxDataDictOut['S']
	d = minmaxDataDictOut['D']
	fig,ax = plt.subplots()

	x1 = np.linspace(m,s)
	yM = (s-x1)/(s-m)
	yS1 = (x1-m)/(s-m) 
	x2 = np.linspace(s,d)
	yS2 = (d-x2)/(d-s)
	yD = (x2-s)/(d-s)

	ax.plot(x1, yM, 'b')
	ax.plot(x1, yS1, 'g')
	ax.plot(x2, yS2, 'g')
	ax.plot(x2, yD, 'y')

	ax.minorticks_on()

	ax.grid(which='major',color = 'k',linewidth = 2)
	ax.grid(which='minor', color = 'k', linestyle = ':')

	plt.xlabel(minmaxDataDictOut['header'], fontsize=14, fontweight='bold')

	plt.annotate('M = '+str(minmaxDataDictOut['M']), xy=(minmaxDataDictOut['M'],1),xycoords='data', xytext = (minmaxDataDictOut['M'],1.03),textcoords='data',fontweight='bold',fontsize=9 )
	plt.annotate('S = '+str(minmaxDataDictOut['S']), xy=(minmaxDataDictOut['S'],1),xycoords='data', xytext = (minmaxDataDictOut['S'],1.03),textcoords='data',fontweight='bold',fontsize=9 )
	plt.annotate('D = '+str(minmaxDataDictOut['D']), xy=(minmaxDataDictOut['D'],1),xycoords='data', xytext = (minmaxDataDictOut['D'],1.03),textcoords='data',fontweight='bold',fontsize=9 )
	if os.path.exists('/home/koza/NL/1/asd/male/Plots/MembFunc/'+minmaxDataDictOut['header']+'Outp.png') == True:
		os.remove('/home/koza/NL/1/asd/male/Plots/MembFunc/'+minmaxDataDictOut['header']+'Outp.png')
	plt.savefig('/home/koza/NL/1/asd/male/Plots/MembFunc/'+minmaxDataDictOut['header']+'Outp.png')
#	plt.show()

value = {'age': 60, 'trestbps':126, 'chol':282, 'thalach':122}
defuzzification(createRegulationBase(['age','trestbps','chol','thalach'],'oldpeak','heart.csv'),value)
createPlotsMembersFunc(searchMinMax('age','heart.csv'))
createPlotsMembersFunc(searchMinMax('trestbps','heart.csv'))
createPlotsMembersFunc(searchMinMax('chol','heart.csv'))
createPlotsMembersFunc(searchMinMax('thalach','heart.csv'))
createPlotsMembersFuncOut(searchMinMaxOut('oldpeak','heart.csv'))
				
				
				
				
		
