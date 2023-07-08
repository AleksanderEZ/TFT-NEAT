---@diagnostic disable: need-check-nil, undefined-global

function GetSprites()
    local sprites = {}
    for i = 0x0200, 0x02FF, 4 do
        local sprite = memory.readbyte(i + 1)
        if USELESS_SPRITES_SET[sprite] ~= true then
            local spriteX = memory.readbyte(i + 3)
            local spriteY = memory.readbyte(i)
            if sprites[spriteX] == nil then sprites[spriteX] = {} end
            sprites[spriteX][spriteY] = true
        end
    end
    for i = 0, 7 do
        local bulletY = memory.readbyte(0x04AA + i)
        local bulletX = memory.readbyte(0x04C3 + i)
        if sprites[bulletX] == nil then sprites[bulletX] = {} end
        sprites[bulletX][bulletY] = true
    end
    return sprites
end

function GetInputs()
    local sprites = GetSprites()
    for X = 0, 255 do
        if sprites[X] ~= nil then
            for Y = 0, 240 do
                if sprites[X][Y] == true then PintarPixel(X, Y, 0xFF00FF00) end
            end
        end
    end
end


function PintarTriangulo()
    gui.drawPolygon({{127, 0},{0, 240},{255, 240}}, 0, 0, 0, 0xFF00FF00)
end

function PintarPixel(X, Y, colorHex)
    gui.drawPixel(X,Y,colorHex)
end


USELESS_SPRITES = {0x30, 0x31, 0x32, 0xFE, 0xFF}
USELESS_SPRITES_SET = {}
POWER_UPS = {0x2D, 0x6F}

for i = 0x00, 0x27 do
    USELESS_SPRITES_SET[i] = true
end

for i = 0xA3, 0xAE do
    USELESS_SPRITES_SET[i] = true
end

for i = 0xD0, 0xED do
    USELESS_SPRITES_SET[i] = true
end

for i = 1, #USELESS_SPRITES do
    USELESS_SPRITES_SET[USELESS_SPRITES[i]] = true
end

while true do
    GetInputs()
    emu.frameadvance()
end