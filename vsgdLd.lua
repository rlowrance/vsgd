-- sgdSZL.lua
-- stochastic gradient descent using the Schaul-Zhang-LeCun procedure to
-- approximate the optimal learning rate and learning rate decay using
-- adaptive component-wise learning rates
-- ref: schaul-12 learning rates.pdf

-- somewhat mimic the stochastic gradient optimization code from Koray
-- ref: https://github.com/koraykv/optim/blob/master/sgd.lua

-- ARGS:
-- opfunc3 : a function defined below
-- theta   : 1D Tensor, initial starting point
-- state   : table with initial values, mutated by this function
--   state.epsilon   : number > 0 ?
--   state.nSamples  : number >= 1, number of samples
--   state.n0        : number >= 1, number of sample to use to kick start
--                     the kick start samples are 1, 2, ..., state.n0
--   state.c         : number > 0, factor used to overestimate variance
--   state.inOrder   : optional boolean, default == false
--                     if true, samples are presented in order 1, 2, ...
--                     if false, a random sample is drawn 

-- where opfunc3 satisfies
-- ARGS: 
-- j           : number, sample number for evaluation
-- RETURNS:
-- f(j)        : number, value of f at sample j
-- gradient(j) : tensor, gradient of f at sample j
-- hessdiag(j) : tensor, the diagonal of the Hessian of f at sample j

-- According to the paper describing the algorithm,
-- The initSamples and c parameters "have only transient initialization
-- effects on the algorith, and thus do not need to be tuned."

-- RETURNS:
-- thetaStar : new theta vector (after one optimization step)
-- sequence  : {f(theta)}
--             one entry table with the function value before the update

-- For now, do not implement the momentum or weight decay features of
-- Koray's code

function sgdSZL(opfunc3, theta, state)

   local trace = false

   if trace then
      print('opfunc3', opfunc3)
      print('theta', theta)
      print('state', state)
   end

   -- type and value check the arguments

   local function assertIsTensor1D(x)
      assert(string.match(torch.typename(x), 'torch%..*Tensor'))
      assert(x:dim() == 1)
   end

   assert(opfunc3)
   assert(type(opfunc3) == 'function')

   assert(theta)
   assertIsTensor1D(theta)
   assert(theta:dim() == 1)

   assert(type(state) == 'table')

   assert(state.epsilon)
   assert(state.epsilon > 0)

   assert(state.nSamples)
   assert(state.nSamples >= 1)
   assert(math.floor(state.nSamples) == state.nSamples) -- is integer

   assert(state.n0)
   assert(state.n0 >= 1)
   assert(math.floor(state.n0) == state.n0)  -- is integer

   assert(state.c)
   assert(state.c > 0)

   -- component wise minimum of scalar and Tensor
   local function min(epsilon, t)
      local n = t:size(1)
      local result = torch.Tensor(n)
      for i = 1, n do
         result[i] = math.min(epsilon, t[i])
      end
      return result
   end

   local n0 = state.nSamples
   local d = theta:size(1) 

   -- initialize g, v, h and tau, if this has not already been done
   if state.g == nil then
      local n0 = state.n0
      local c = state.c
      
      state.tau = torch.Tensor(d):fill(n0)

      -- compute all i = 1,d values at once
      local sumGradients = torch.Tensor(d):fill(0)
      local sumVariances = torch.Tensor(d):fill(0)
      local sumHessdiags = torch.Tensor(d):fill(0)
      for j = 1, n0 do
         local f, gradient, hessdiag = opfunc3(j)
         
         if j == 1 then
            -- type check opfunc3 returned values only on the first time
            assert(f)
            assert(type(f) == 'number')
            
            assert(gradient)
            assertIsTensor1D(gradient)
            
            assert(hessdiag)
            assertIsTensor1D(hessdiag)
         end
         
         sumGradients = torch.add(sumGradients,
                                  gradient)
         sumVariances = torch.add(sumVariances, 
                                  torch.cmul(gradient, gradient))
         sumHessdiags = torch.add(sumHessdiags, 
                                  hessdiag)
      end
      
      if trace then
         print('sumGradients', sumGradients)
         print('sumVariances', sumVariances)
         print('sumHessdiags', sumHessdiags)
         print('n0', n0)
         print('c', state.c)
      end
   
      state.g = sumGradients / n0
      state.v = (sumVariances / n0) * c
      state.h = min(state.epsilon, (sumHessdiags / n0) * c)
      
      if trace then
         print('initialized values')
         print('g', state.g)
         print('v', state.v)
         print('h', state.h)
      end
      end -- first-time initialization

   -- perform update:
   -- 1. draw sample
   -- 2. compute gradient and diagnonal hessian
   -- 3. update moving averages of g, v, h
   -- 4. estimate new best learning rate
   -- 5. take a stochastic gradient step
   -- 6. update memory size

   -- 1. draw random integer in [1, init.nSamples], or 
   --    process samples in order
   if state.inOrder ~= nil then
      state.j = (state.j or 0) + 1
      if state.j > state.nSamples then
         state.j = 1
      end
   else
      state.j = (torch.random() % state.nSamples) + 1
   end

   if trace then
      print('sample index j', state.j)
   end

   -- 2. compute gradient and diagonal hessian
   f, gradient, hessdiag = opfunc3(state.j)

   -- 3. update moving average of g, v, h
   local one = torch.Tensor(d):fill(1)
   local oneOverTau = torch.cdiv(one, state.tau)
   local oneMinusOneOverTau = one - oneOverTau

   state.g = 
      torch.cmul(oneMinusOneOverTau, state.g) + torch.cmul(oneOverTau, gradient)

   state.v = 
      torch.cmul(oneMinusOneOverTau, state.v) + torch.cmul(oneOverTau,
                                                           torch.cmul(gradient,
                                                                      gradient))
   state.h = 
      torch.cmul(oneMinusOneOverTau, state.h) + torch.cmul(oneOverTau,
                                                           min(state.epsilon,

                                                               hessdiag))

   -- 4. estimate new best learning rate
   local gg = torch.cmul(state.g, state.g)
   local hv = torch.cmul(state.h, state.v)
   state.eta = torch.cdiv(gg, hv)

   -- 5. take a stochastic gradient step
   theta = theta:add(-state.eta, gradient)
      
   -- 6. update memory size

   state.tau = 
      torch.cmul(one - torch.cdiv(gg, state.v), state.tau) + one

   return theta, {f}
end -- function sgdSZL
