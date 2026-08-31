function otf = psf2otf_gpu(varargin)
%PSF2OTF Convert point-spread function to optical transfer function.
%   OTF = PSF2OTF(PSF) computes the Fast Fourier Transform (FFT) of the
%   point-spread function (PSF) array and creates the optical transfer
%   function (OTF) array that is not influenced by the PSF off-centering. By
%   default, the OTF array is the same size as the PSF array.
% 
%   OTF = PSF2OTF(PSF,OUTSIZE) converts the PSF array into an OTF array of
%   specified size OUTSIZE. The OUTSIZE cannot be smaller than the PSF
%   array size in any dimension.
%
%   To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
%   post-pads the PSF array (down or to the right) with zeros to match
%   dimensions specified in OUTSIZE, then circularly shifts the values of
%   the PSF array up (or to the left) until the central pixel reaches (1,1)
%   position.
%
%   Note that this function is used in image convolution/deconvolution 
%   when the operations involve the FFT. 
%
%   Class Support
%   -------------
%   PSF can be any nonsparse, numeric array, including gpuArray. OTF will be
%   of the same type as PSF.
%
%   Example
%   -------
%      PSF  = fspecial('gaussian',13,1);
%      OTF  = psf2otf(PSF,[31 31]); % PSF --> OTF
%      subplot(1,2,1); surf(PSF); title('PSF');
%      axis square; axis tight
%      subplot(1,2,2); surf(abs(OTF)); title('corresponding |OTF|');
%      axis square; axis tight
%
%   See also OTF2PSF, CIRCSHIFT, PADARRAY.

% 允许gpuArray输入
[psf, psfSize, outSize] = ParseInputs(varargin{:});

% 检查是否所有PSF值都为零或PSF为空
if isempty(psf)
    allZeros = true;
else
    allZeros = all(psf(:) == 0);
    % 如果是gpuArray，需要将结果提取到CPU
    if isa(psf, 'gpuArray')
        allZeros = gather(allZeros);
    end
end

if ~allZeros
   % Pad the PSF to outSize
   padSize = outSize - psfSize;
   psf = padarray(psf, padSize, 'post');

   % Circularly shift otf so that the "center" of the PSF is at the
   % (1,1) element of the array.
   psf = circshift(psf, -floor(psfSize/2));

   % Compute the OTF
   otf = fftn(psf);

   % Estimate the rough number of operations involved in the 
   % computation of the FFT.
   nElem = prod(psfSize);
   nOps = 0;
   for k=1:ndims(psf)
      nffts = nElem/psfSize(k);
      nOps = nOps + psfSize(k)*log2(psfSize(k))*nffts; 
   end

   % 计算虚部与实部的比例
   maxAbs = max(abs(otf(:)));
   maxImag = max(abs(imag(otf(:))));
   
   % 如果是gpuArray，需要提取到CPU进行比较
   if isa(otf, 'gpuArray')
       maxAbs = gather(maxAbs);
       maxImag = gather(maxImag);
   end
   
   % Discard the imaginary part of the psf if it's within roundoff error,
   % or if maxAbs is extremely small
   if maxAbs < eps || (maxAbs > 0 && maxImag/maxAbs <= nOps*eps)
      otf = real(otf);
   end
else
   % 使用与输入类型匹配的零数组
   otf = zeros(outSize, 'like', psf);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Parse inputs
%%%

function [psf, psfSize, outSize] = ParseInputs(varargin)

narginchk(1,2)
switch nargin
case 1       % PSF2OTF(PSF) 
  psf = varargin{1};   
case 2       % PSF2OTF(PSF,OUTSIZE) 
  psf = varargin{1}; 
  outSize = varargin{2};
end

% Check validity of the input parameters
% psf can be empty. it treats empty array as the fftn does
if ~isnumeric(psf) || issparse(psf)
  error(message('images:psf2otf:expectedNonSparseAndNumeric'))
else
    % 确保psf是double，但保持其是否为gpuArray
    if ~isa(psf, 'double')
        psf = double(psf);
    end
  
  % 检查是否有限
  if ~isempty(psf)
      allFinite = all(isfinite(psf(:)));
      if isa(psf, 'gpuArray')
          allFinite = gather(allFinite);
      end
      if ~allFinite
        error(message('images:psf2otf:expectedFinite'))
      end
  end
end
psfSize = size(psf);

% outSize:
if nargin==1
  outSize = psfSize;% by default
elseif ~isa(outSize, 'double')
  error(message('images:psf2otf:invalidType'))
elseif any(outSize(:)<0) || ~isreal(outSize) || ...
    all(size(outSize)>1) || ~all(isfinite(outSize(:)))
  error(message('images:psf2otf:invalidOutSize'))
end

if isempty(outSize)
  outSize = psfSize;
elseif ~isempty(psf) % empty arrays are treated similar as in the fftn
  % padlength函数的实现
  [psfSize, outSize] = padlength(psfSize, outSize(:).');
  if any(outSize < psfSize)
    error(message('images:psf2otf:outSizeIsSmallerThanPsfSize'))
  end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Pad length helper function
%%%

% 辅助函数
function [sizeI, sizePSF] = padlength(sizeI, sizePSF)
    lI = length(sizeI);
    lP = length(sizePSF);
    if lI > lP
        sizePSF = [sizePSF ones(1, lI-lP)];
    elseif lI < lP
        sizeI = [sizeI ones(1, lP-lI)];
    end
end