-- Vsgd.lua
-- stochastic gradient descent using the Schaul-Zhang-LeCun procedure to
-- approximate the optimal learning rate and learning rate decay using
-- adaptive component-wise learning rates.

-- ref: schaul-12 learning rates.pdf. This code implements algorithm 1 in 
-- the paper

-- mimic the stochastic gradient optimization code from Koray
-- ref: https://github.com/koraykv/optim/blob/master/sgd.lua


local Vsgd = torch.class('Vsgd')

--------------------------------------------------------------------------------
-- _init
--------------------------------------------------------------------------------

function Vsgd:__init()
end

--------------------------------------------------------------------------------
-- ld
--------------------------------------------------------------------------------

-- take one iteration step from current or initial position using
-- the local variance and local diagonal Hessian estimates
-- ARGS:
-- opfunc3 : a function defined below
-- theta   : 1D Tensor, initial starting point
-- state   : table with initial values, mutated by this function
--   state.epsilon   : number > 0 ?
--   state.n0        : number >= 1, number of sample to use to kick start
--   state.c         : number > 0, factor used to overestimate variance

-- where opfunc3 is a function of the parameters and possible an implicit
-- sample that is somehow selects
-- and returns 3 values:
-- fx              : numer, value of f at theta and the sample
-- gradient(theta) : tensor, gradient of f wrt theta at theta and the sample
-- hessdiag(theta) : tensor, the diagonal of the Hessian of f at theta and
--                   the sample

-- The first n0 calls to f should evaluate f at the n0 initial samples

-- According to the paper describing the algorithm,
-- The initSamples and c parameters "have only transient initialization
-- effects on the algorithm, and thus do not need to be tuned." However,
-- experiments show that too low a value for c can lead to NaN results.

-- RETURNS:
-- thetaStar : new theta vector (after one optimization step)
-- sequence  : {opfunc(theta)}
--             one entry table with the function value before the update

function Vsgd:ld(f, theta, state)

   local trace = true

   if trace then
      print('f', f)
      print('theta') print(theta)
      print('state') print( state)
   end

   -- type and value check the arguments on first call only

   if state.tau == nil then
      -- since first time called, type and value check the parameters
      self:_typeAndValueCheck(f, theta, state)
   end
   
   local n0 = state.n0
   local d = theta:size(1) 

   -- initialize g, v, h and tau, if this has not already been done
   if state.tau == nil then
      self:_initializeState(f, state, theta)
      if trace then
         print('state after initialization')
         self:printState(state)
      end
   end

   -- perform update:
   -- 1. draw sample
   -- 2. compute gradient and diagnonal hessian
   -- 3. update moving averages of g, v, h
   -- 4. estimate new best learning rate
   -- 5. take a stochastic gradient step
   -- 6. update memory size

   -- 1. draw a sample: the function f does this on its own, so nothing to do

   -- 2. compute gradient and diagonal hessian
   local fx, gradient, hessdiag = f(theta)
   if trace then
      print(' gradient') print(gradient)
      print(' hessdiag') print(hessdiag)
   end

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
      
   state.h =  -- TODO: take abs value!
      torch.cmul(oneMinusOneOverTau, state.h) + 
      torch.cmul(oneOverTau,
                 self:_max(state.epsilon,
                           torch.abs(hessdiag)))

   if trace then
      print('updated moving averages')
      print('g') print(state.g)
      print('v') print(state.v)
      print('h') print(state.h)
   end

   -- 4. estimate new best learning rate
   local gg = torch.cmul(state.g, state.g)
   local hv = torch.cmul(state.h, state.v)
   state.eta = torch.cdiv(gg, hv)
   if trace then
      print('gg') print(gg)
      print('hv') print(hv)
      print('eta') print(state.eta)
   end
   -- replace any NaN values with zero
   for i = 1, d do
      if state.eta[i] ~= state.eta[i] then
         -- the i-th coordinate is NaN
         state.eta[i] = 0
      end
   end

   -- 5. take a stochastic gradient step to update the parameter
   if trace then
      print('theta before step') print(theta)
      print('eta') print(state.eta)
      print('gradient') print( gradient)
   end

   theta = theta - torch.cmul(state.eta, gradient)
   --theta = theta:add(-state.eta, gradient)

   if trace then
      print('theta after step') print(theta)
      print('tau before update') print(state.tau)
      --halt()
   end

      
   -- 6. update memory size

   state.priorTau = state.tau
   state.tau = 
      torch.cmul(one - torch.cdiv(gg, state.v), state.tau) + one
   if trace then
      print('updated tau') print(state.tau)
   end
   -- replace any NaN values with previous tau values
   for i = 1, d do
      if state.tau[i] ~= state.tau[i] then
         -- the i-th coordinate is NaN
         state.tau[i] = state.priorTau[i]
         if trace then
            print('correcting tau coordinate', i)
         end
      end
   end

   return theta, {fx}
end -- ld

--------------------------------------------------------------------------------
-- printState
--------------------------------------------------------------------------------

-- nicely print the state fields
function Vsgd:printState(state)
   assert(state)

   local function maybePrint(name, value)
      if value == nil then return end
      local formatter = ' %10s = %s'
      if type(value) == 'number' then
         print(string.format(formatter, name, tostring(value)))
         return
      end
      -- assume value is a torch.Tensor
      s = '['
      for i = 1, value:size(1) do
         if i ~= 1 then s = s .. ', ' end
         s = s .. tostring(value[i])
      end
      s = s .. ']'
      print(string.format(formatter, name, s))
   end
   
   print('Vsgd state argument')
   maybePrint('c', state.c)
   maybePrint('epsilon', state.epsilon)
   maybePrint('eta', state.eta)
   maybePrint('g', state.g)
   maybePrint('h', state.h)
   maybePrint('n0', state.n0)
   maybePrint('tau', state.tau)
   maybePrint('v', state.v)
end -- printState

--------------------------------------------------------------------------------
-- _initializeState
--------------------------------------------------------------------------------

-- initalize g, v, h, tau
function Vsgd:_initializeState(f, state, theta)
   assert(f)
   assert(state)
   assert(theta)

   local c = state.c
   local n0 = state.n0
   local d = theta:size(1)

   state.tau = torch.Tensor(d):fill(n0)
   
   -- compute all i = 1,d values at once
   local sumGradients = torch.Tensor(d):fill(0)
   local sumVariances = torch.Tensor(d):fill(0)
   local sumHessdiags = torch.Tensor(d):fill(0)
   for j = 1, n0 do
      -- f must be written to iterate over the first n0 samples
      local fx, gradient, hessdiag = f(theta)
      
      if j == 1 then
         -- type check f's returned values only on the first time
         assert(gradient)
         self:_assertIsTensor1D(gradient)
         
         assert(hessdiag)
         self:_assertIsTensor1D(hessdiag)
      end
      
      sumGradients = torch.add(sumGradients, gradient)
      sumVariances = torch.add(sumVariances, torch.cmul(gradient, gradient))
      sumHessdiags = torch.add(sumHessdiags, hessdiag)
   end
   
   if trace then
      print('sumGradients', sumGradients)
      print('sumVariances', sumVariances)
      print('sumHessdiags', sumHessdiags)
      print('n0', state.n0)
      print('c', state.c)
   end
   
   state.g = sumGradients / n0
   state.v = (sumVariances / n0) * c
   state.h = self:_max(state.epsilon, (sumHessdiags / n0) * c)
   
   if trace then
      print('initialized values')
      print('g', state.g)
      print('v', state.v)
      print('h', state.h)
   end
end -- _initializeState

--------------------------------------------------------------------------------
-- _max
--------------------------------------------------------------------------------

-- return component-wise maximum of scalar and Tensor
function Vsgd:_max(epsilon, t)
   local n = t:size(1)
   local result = torch.Tensor(n)
   for i = 1, n do
      result[i] = math.max(epsilon, t[i])
   end
   return result
end -- _max

-------------------------------------------------------------------------------
-- _assertIsTensor1D
------------------------------------------------------------------------------- 

function Vsgd:_assertIsTensor1D(x)
   assert(string.match(torch.typename(x), 'torch%..*Tensor'))
   assert(x:dim() == 1)
end


--------------------------------------------------------------------------------
-- _typeAndValueCheck
--------------------------------------------------------------------------------

-- type and value check the parameters
function Vsgd:_typeAndValueCheck(f, theta, state)
   -- f is a function
   assert(f)
   assert(type(f) == 'function')
   
   -- theta is a 1D Tensor
   assert(theta)
   self:_assertIsTensor1D(theta)
   assert(theta:dim() == 1)
   
   -- state is a table
   assert(type(state) == 'table')
   
   -- state.eplison is a positive number
   assert(state.epsilon)
   assert(state.epsilon > 0)
   
   -- state.n0 is a positive integer
   assert(state.n0)
   assert(state.n0 >= 1)
   assert(math.floor(state.n0) == state.n0)  -- is integer
   
   -- state.c is a positive number
   assert(state.c)
   assert(state.c > 0)
end -- _typeAndValueCheck

