#!/bin/bash

# Set path



START=$(date +%s)

ANTsPATH=$5

# fixname is the one will be warped
fixname=$1
movname=$2
inputPath=$3
outputPath=$4

maskfixLow=${inputPath}/${fixname}_m.nii.gz
maskmovLow=${inputPath}/${movname}_m.nii.gz
ctImagefixLow=${inputPath}/${fixname}.nii.gz
ctImagemovLow=${inputPath}/${movname}.nii.gz

outputPrefix=${outputPath}/${fixname}_To_${movname}_outputPrefixResults

# check directory
if [ ! -d ${outputPath} ]
  then 
  mkdir -p ${outputPath}
fi

echo "Starting Mask Registration"
${ANTsPATH}/antsRegistration -d 3 \
                 -o ${outputPrefix} \
                 -r [${maskfixLow},${maskmovLow},1] \
                 -t Rigid[0.2] \
                 -m MI[${maskfixLow},${maskmovLow},1,8,Regular,0.15] \
                 -c 100x100x50 \
                 -s 4x2x1 \
                 -f 8x6x4 \
                 -t Affine[0.2] \
                 -m MI[${maskfixLow},${maskmovLow},1,8,Regular,0.15] \
                 -c 100x100x50 \
                 -s 4x2x1 \
                 -f 8x6x4 \
                 -t BSplineSyN[0.2,40,0] \
                 -m MSQ[${maskfixLow},${maskmovLow},1,1] \
                 -c 100x50x0 \
                 -s 2x1x0 \
                 -f 4x2x1 \

END=$(date +%s)
DIFF=$(( $END - $START ))
echo "Mask took $DIFF seconds"


# Here we dilate the masks 
maskfixDilated=${inputPath}/${fixname}_mLow_Dilated.nii.gz
maskmovDilated=${inputPath}/${movname}_mLow_Dilated.nii.gz

${ANTsPATH}/ImageMath 3 ${maskfixDilated} MD ${maskfixLow} 20 # was 10 
${ANTsPATH}/ImageMath 3 ${maskmovDilated} MD ${maskmovLow} 20 # was 10

${ANTsPATH}/antsRegistration -d 3 \
                 -o ${outputPrefix} \
                 -r ${outputPrefix}1Warp.nii.gz \
                 -r ${outputPrefix}0GenericAffine.mat \
                 -t BSplineSyN[0.2,14,0] \
                 -m CC[${ctImagefixLow},${ctImagemovLow},1,4] \
                 -c 50x20x5 \
                 -s 2x1x0 \
                 -f 4x2x1 \
                 -x [${maskfixDilated},${maskmovDilated}]


fix2movWarped=${outputPath}/${fixname}_To_${movname}_Warped.nii.gz

${ANTsPATH}/antsApplyTransforms -d 3 \
                               -o $fix2movWarped \
                               -n Linear \
                               -r ${ctImagemovLow} \
                               -i ${ctImagefixLow} \
                               -t [${outputPath}/${fixname}_To_${movname}_outputPrefixResults0GenericAffine.mat,1] \
                               -t ${outputPath}/${fixname}_To_${movname}_outputPrefixResults1InverseWarp.nii.gz \
                               -t ${outputPath}/${fixname}_To_${movname}_outputPrefixResults2InverseWarp.nii.gz

# First compose the affine and deformable transforms into a single warp field
${ANTsPATH}/antsApplyTransforms -d 3 \
                                    -r ${ctImagemovLow} \
                                    -o [${outputPath}/${fixname}_To_${movname}_TotalWarp.nii.gz,1] \
                                    -t [${outputPath}/${fixname}_To_${movname}_outputPrefixResults0GenericAffine.mat,1] \
                                    -t ${outputPath}/${fixname}_To_${movname}_outputPrefixResults1InverseWarp.nii.gz \
                                    -t ${outputPath}/${fixname}_To_${movname}_outputPrefixResults2InverseWarp.nii.gz

${ANTsPATH}/CreateJacobianDeterminantImage 3 \
      ${outputPath}/${fixname}_To_${movname}_TotalWarp.nii.gz  \
      ${outputPath}/${fixname}_To_${movname}_Jacobian.nii.gz 0 1

# TotalWarp above is defined on movname's grid and points movname->fixname
# (that's what resampling fixname onto movname needs). Also compose the
# forward chain into a field defined on fixname's own grid, pointing
# fixname->movname (i.e. literally "from fixname to movname").
${ANTsPATH}/antsApplyTransforms -d 3 \
                                    -r ${ctImagefixLow} \
                                    -o [${outputPath}/${fixname}_To_${movname}_TotalWarp_Forward.nii.gz,1] \
                                    -t ${outputPath}/${fixname}_To_${movname}_outputPrefixResults2Warp.nii.gz \
                                    -t ${outputPath}/${fixname}_To_${movname}_outputPrefixResults1Warp.nii.gz \
                                    -t ${outputPath}/${fixname}_To_${movname}_outputPrefixResults0GenericAffine.mat

END=$(date +%s)
DIFF=$(( $END - $START ))
echo "Registration took $DIFF seconds"

# Done
